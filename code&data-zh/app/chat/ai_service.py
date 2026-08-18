"""
AI 服务模块，集成意图识别和问答功能
"""
import logging
from typing import List, Dict
from openai import OpenAI
import json

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化客户端
client = OpenAI(api_key="yours", base_url="https://api.deepseek.com")

# Few-shot 提示词
prompt_Few_shot = """Q:我的样品是贵金属负载型催化剂，我想研究它的电子价态和配位环境，我应该选择哪种同步辐射技术？它们有什么区别 
A:电子价态和配位环境
Q:我应该用什么软件去处理XAFS原始数据呢 
A:处理XAFS原始数据的软件 
Q:我想测Pd K边的吸收谱，某某线站的能量范围可以满足要求吗 
A:Pd K边的吸收谱线站的能量范围
"""


def _script_counts(text: str) -> Dict[str, int]:
    counts = {
        "chinese": 0,
        "japanese": 0,
        "korean": 0,
        "latin": 0,
        "cyrillic": 0,
        "arabic": 0,
        "devanagari": 0,
        "thai": 0,
        "greek": 0,
        "hebrew": 0,
    }
    for char in text or "":
        code = ord(char)
        if char.isspace() or char.isdigit():
            continue
        if 0x3040 <= code <= 0x30FF:
            counts["japanese"] += 1
        elif 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF:
            counts["chinese"] += 1
        elif 0xAC00 <= code <= 0xD7AF or 0x1100 <= code <= 0x11FF:
            counts["korean"] += 1
        elif "A" <= char <= "Z" or "a" <= char <= "z" or 0x00C0 <= code <= 0x024F:
            counts["latin"] += 1
        elif 0x0400 <= code <= 0x052F:
            counts["cyrillic"] += 1
        elif 0x0600 <= code <= 0x06FF:
            counts["arabic"] += 1
        elif 0x0900 <= code <= 0x097F:
            counts["devanagari"] += 1
        elif 0x0E00 <= code <= 0x0E7F:
            counts["thai"] += 1
        elif 0x0370 <= code <= 0x03FF:
            counts["greek"] += 1
        elif 0x0590 <= code <= 0x05FF:
            counts["hebrew"] += 1
    return counts


def _dominant_script(text: str) -> str:
    counts = _script_counts(text)
    if counts["japanese"] > 0:
        return "japanese"
    script, count = max(counts.items(), key=lambda item: item[1])
    return script if count else "unknown"


def _is_likely_english(text: str) -> bool:
    lowered = f" {text.lower()} "
    english_markers = (
        " what ", " which ", " how ", " why ", " when ", " where ", " should ",
        " can ", " could ", " would ", " is ", " are ", " do ", " does ",
        " please ", " recommend ", " sample ", " experiment ", " beamline ",
    )
    return _dominant_script(text) == "latin" and any(marker in lowered for marker in english_markers)


def _has_meaningful_script(text: str, script: str, minimum: int = 8) -> bool:
    return _script_counts(text).get(script, 0) >= minimum


def _latin_language_instruction(text: str) -> str:
    lowered = f" {text.lower()} "
    if _is_likely_english(text):
        return "English"
    if any(word in lowered for word in (" que ", " cuál ", " cuales ", " cómo ", " por qué ", " muestra ", " experimento ")):
        return "Spanish"
    if any(word in lowered for word in (" quel ", " quelle ", " comment ", " pourquoi ", " échantillon ", " expérience ")):
        return "French"
    if any(word in lowered for word in (" welche ", " was ", " wie ", " warum ", " probe ", " experiment ")):
        return "German"
    if any(word in lowered for word in (" qual ", " como ", " por que ", " amostra ", " experimento ")):
        return "Portuguese"
    if any(word in lowered for word in (" quale ", " come ", " perché ", " campione ", " esperimento ")):
        return "Italian"
    return "the same Latin-script language as the user's input"


def _needs_language_alignment(user_text: str, response_text: str) -> bool:
    user_script = _dominant_script(user_text)
    response_script = _dominant_script(response_text)
    if user_script == "unknown" or response_script == "unknown":
        return False

    if _is_likely_english(user_text):
        # Knowledge-base answers often mix English terms with Chinese prose.
        # If an English question receives any meaningful Chinese body text,
        # force translation even when Latin technical terms dominate the count.
        return _has_meaningful_script(response_text, "chinese")

    if user_script == "chinese":
        return response_script != "chinese"

    if user_script != response_script:
        return True

    # For non-English Latin languages, keep one alignment pass so English drafts
    # can still be converted to the user's exact language.
    return user_script == "latin"


def _target_language_instruction(user_text: str) -> str:
    script = _dominant_script(user_text)
    if script == "chinese":
        return "Simplified Chinese"
    if script == "japanese":
        return "Japanese"
    if script == "korean":
        return "Korean"
    if script == "cyrillic":
        return "the same Cyrillic-script language as USER_INPUT"
    if script == "arabic":
        return "Arabic"
    if script == "devanagari":
        return "the same Devanagari-script language as USER_INPUT"
    if script == "thai":
        return "Thai"
    if script == "greek":
        return "Greek"
    if script == "hebrew":
        return "Hebrew"
    if script == "latin":
        return _latin_language_instruction(user_text)
    return "the same language as the user's input"


def language_follow_instruction(user_text: str) -> str:
    target_language = _target_language_instruction(user_text)
    return (
        f"Answer in {target_language}. Keep the answer language consistent with "
        "the user's input language. Preserve professional terms, beamline names, "
        "abbreviations, numbers, and units."
    )


def _looks_like_user_question_instead_of_answer(user_text: str, aligned_text: str) -> bool:
    user_words = {word.strip(".,?!:;()[]{}'\"").lower() for word in user_text.split()}
    aligned_words = {word.strip(".,?!:;()[]{}'\"").lower() for word in aligned_text.split()}
    user_words = {word for word in user_words if len(word) > 3}
    aligned_words = {word for word in aligned_words if len(word) > 3}
    if not user_words or not aligned_words:
        return False
    overlap = len(user_words & aligned_words) / max(len(user_words), 1)
    question_like = "?" in aligned_text or aligned_text.strip().lower().startswith(
        ("what ", "which ", "how ", "why ", "when ", "where ", "can ", "should ")
    )
    return question_like and overlap >= 0.45


def wd(que: str) -> str:
    try:
        stream = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {"role": "system", "content": "你是一个同步辐射实验助手，根据文本回答用户问题。" + language_follow_instruction(que)},
                {"role": "user", "content": que}
            ],
            stream=False
        )
        return stream.choices[0].message.content
    except Exception as e:
        logger.error(f"DeepSeek 调用错误: {e}")
        return "抱歉，我遇到了一些问题，请稍后再试。"


def wd_vector(que: str, s: str) -> str:
    try:
        stream = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {"role": "system", "content": s + "\n" + language_follow_instruction(que)},
                {"role": "user", "content": que}
            ],
            stream=False
        )
        return stream.choices[0].message.content
    except Exception as e:
        logger.error(f"DeepSeek 调用错误: {e}")
        return "抱歉，我遇到了一些问题，请稍后再试。"


def multilingual_input(que: str) -> str:
    try:
        stream = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {"role": "system",
                 "content": "你是一个精通多国语言的翻译智能体。如果下面内容是中文直接输出原内容。如果下面内容不是中文，把下面用户内容内容翻译为中文。"},
                {"role": "user", "content": que}
            ],
            stream=False
        )
        return stream.choices[0].message.content
    except Exception as e:
        logger.error(f"DeepSeek 调用错误: {e}")
        return "抱歉，我遇到了一些问题，请稍后再试。"


def multilingual_output(que: str, response_result: str) -> str:
    if not _needs_language_alignment(que, response_result):
        return response_result

    target_language = _target_language_instruction(que)
    protected_answer = f"<ANSWER_TO_TRANSLATE>\n{response_result}\n</ANSWER_TO_TRANSLATE>"
    try:
        stream = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict answer translator. Translate only the text inside "
                        f"<ANSWER_TO_TRANSLATE> into {target_language}. Do not answer the user's "
                        "question. Do not translate or repeat the user's question. Do not add "
                        "new facts. Preserve technical terms, beamline names, abbreviations, "
                        "numbers, and units. Return only the translated answer text, without tags."
                    ),
                },
                {
                    "role": "user",
                    "content": protected_answer,
                }
            ],
            stream=False
        )
        aligned_response = stream.choices[0].message.content
        if _looks_like_user_question_instead_of_answer(que, aligned_response):
            logger.warning("语言对齐结果疑似返回了用户问题，保留原始回答")
            return response_result
        return aligned_response
    except Exception as e:
        logger.error(f"DeepSeek 调用错误: {e}")
        return response_result


def intent(que: str) -> str:
    try:
        stream = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {"role": "system",
                 "content": "你是一个用户问题意图识别专家，不需要回答问题，只需要按照例子进行意图提取，保留完整信息" + "\n" + prompt_Few_shot},
                {"role": "user", "content": 'Q:' + que}
            ],
            stream=False,
            temperature=0
        )
        return stream.choices[0].message.content
    except Exception as e:
        logger.error(f"意图识别 调用错误: {e}")
        return que  # 如果意图识别失败，返回原问题


with open("./data/BSRF_HEPS_Experimental_Techniques.json", "r",
          encoding="utf-8") as f:
    kg_data = json.load(f)


def question_classifer(que):
    try:
        stream = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {"role": "system",
                 "content": "判断问题是否能在知识图谱中找到答案，能的话输出“yes”，不能的话输出“no”，请实事求是。图谱内容如下：" + str(
                     kg_data)},
                {"role": "user", "content": que}
            ],
            stream=False
            # max_tokens=4096,
            # temperature=0.7
        )
        s = ""

        s += stream.choices[0].message.content

        return s
    except Exception as e:
        return logger.error(f"问题分类 错误: {e}")


def wd_kg_techniques(que):
    try:

        stream = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {"role": "system",
                 "content": "你是一个北京同步辐射装置和高能同步辐射装置实验助手，根据参考内容针对性的回答用户问题，要求简洁和总结性。" + str(
                     kg_data) + "\n" + language_follow_instruction(que)},
                {"role": "user", "content": que}
            ],
            stream=False
            # max_tokens=4096,
            # temperature=0.7
        )
        s = ""

        s += stream.choices[0].message.content

        return s
    except Exception as e:
        return logger.error(f"根据知识图谱技术类问答 错误: {e}")


def get_chat_messages(conversation_messages: List[Dict], user_message: str = "") -> List[Dict]:
    """
    将数据库中的消息转换为 API 格式
    
    参数:
        conversation_messages: 会话中的消息列表
        
    返回:
        API 格式的消息列表
    """
    messages = [
        {"role": "system", "content": "你是一个北京同步辐射装置和高能同步辐射装置的助手，根据文本回答用户问题。" + language_follow_instruction(user_message)}
    ]

    for msg in conversation_messages:
        messages.append({
            "role": msg['role'],
            "content": msg['content']
        })

    return messages


def chat_with_history(user_message: str, conversation_messages: List[Dict]) -> str:
    """
    基于历史消息进行多轮对话
    
    参数:
        user_message: 用户当前消息
        conversation_messages: 历史消息列表
        
    返回:
        助手回复
    """
    try:
        # 构建消息列表
        messages = get_chat_messages(conversation_messages, user_message)
        messages.append({"role": "user", "content": user_message})

        stream = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=messages,
            stream=False
        )
        return stream.choices[0].message.content
    except Exception as e:
        logger.error(f"多轮对话 调用错误: {e}")
        return "抱歉，我遇到了一些问题，请稍后再试。"
