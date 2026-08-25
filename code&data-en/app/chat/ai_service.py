"""
AI Service module, integrating intent recognition and question and answer functions
"""
import logging
from typing import List, Dict
from openai import OpenAI
import json
from django.conf import settings

# Configuration log
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_deepseek_client() -> OpenAI:
    """Create a DeepSeek client from environment-backed Django settings."""
    if not settings.DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY is not configured.")
    return OpenAI(
        api_key=settings.DEEPSEEK_API_KEY,
        base_url=settings.DEEPSEEK_BASE_URL,
    )

# Few-shot prompt word
prompt_Few_shot = """Q:My sample is a precious metal supported catalyst and I want to study its electronic valence state and coordination environment. Which synchrotron radiation technology should I choose? What is the difference between them
A:Electronic valence state and coordination environment
Q:What software should I use to process it?XAFSWhat about the original data?
A:ProcessXAFSRaw data software
Q:I want to testPd KCan the edge absorption spectrum and the energy range of a certain line station meet the requirements?
A:Pd KThe energy range of the edge absorption line station
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
        stream = get_deepseek_client().chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {"role": "system", "content": "You are a synchrotron radiation experiment assistant, answering user questions based on text." + language_follow_instruction(que)},
                {"role": "user", "content": que}
            ],
            stream=False
        )
        return stream.choices[0].message.content
    except Exception as e:
        logger.error(f"DeepSeek Call error: {e}")
        return "Sorry, I encountered some problems, please try again later."


def wd_vector(que: str, s: str) -> str:
    try:
        stream = get_deepseek_client().chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {"role": "system", "content": s + "\n" + language_follow_instruction(que)},
                {"role": "user", "content": que}
            ],
            stream=False
        )
        return stream.choices[0].message.content
    except Exception as e:
        logger.error(f"DeepSeek Call error: {e}")
        return "Sorry, I encountered some problems, please try again later."


def multilingual_input(que: str) -> str:
    try:
        stream = get_deepseek_client().chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {"role": "system",
                 "content": "You are a translation agent proficient in multiple languages. If the following content is in Chinese, directly output the original content. If the content below is not in Chinese, translate the user content below into Chinese."},
                {"role": "user", "content": que}
            ],
            stream=False
        )
        return stream.choices[0].message.content
    except Exception as e:
        logger.error(f"DeepSeek Call error: {e}")
        return "Sorry, I encountered some problems, please try again later."


def multilingual_output(que: str, response_result: str) -> str:
    if not _needs_language_alignment(que, response_result):
        return response_result

    target_language = _target_language_instruction(que)
    protected_answer = f"<ANSWER_TO_TRANSLATE>\n{response_result}\n</ANSWER_TO_TRANSLATE>"
    try:
        stream = get_deepseek_client().chat.completions.create(
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
            logger.warning("The language alignment results are suspected of returning user questions, and the original answers are retained.")
            return response_result
        return aligned_response
    except Exception as e:
        logger.error(f"DeepSeek Call error: {e}")
        return response_result


def intent(que: str) -> str:
    try:
        stream = get_deepseek_client().chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {"role": "system",
                 "content": "You are an expert in user question intent recognition. You do not need to answer questions. You only need to follow examples to extract intent and retain complete information." + "\n" + prompt_Few_shot},
                {"role": "user", "content": 'Q:' + que}
            ],
            stream=False,
            temperature=0
        )
        return stream.choices[0].message.content
    except Exception as e:
        logger.error(f"Intent recognition calling error: {e}")
        return que  # If intent recognition fails, return to the original question


with open(settings.TECHNIQUE_KG_PATH, "r", encoding="utf-8") as f:
    kg_data = json.load(f)


def question_classifier(que):
    try:
        stream = get_deepseek_client().chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {"role": "system",
                 "content": "Determine whether the answer can be found in the knowledge graph. Output only 'yes' or 'no'. The knowledge graph content follows:" + str(
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
        return logger.error(f"Problem classification error: {e}")


# Backward-compatible alias for callers using the original function name.
question_classifer = question_classifier


def wd_kg_techniques(que):
    try:

        stream = get_deepseek_client().chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {"role": "system",
                 "content": "You are an experimental assistant at the Beijing Synchrotron Radiation Facility and the High-Energy Synchrotron Radiation Facility. You answer user questions in a targeted manner based on the reference content and are required to be concise and summarized." + str(
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
        return logger.error(f"Questions and answers based on knowledge graph technical questions Error: {e}")


def get_chat_messages(conversation_messages: List[Dict], user_message: str = "") -> List[Dict]:
    """
    Convert the message in the database toAPI Format
    
    parameters:
        conversation_messages: List of messages in conversation
        
    Return:
        API Format message list
    """
    messages = [
        {"role": "system", "content": "You are an assistant at Beijing Synchrotron Radiation Facility and High Energy Synchrotron Radiation Facility, answering user questions based on text." + language_follow_instruction(user_message)}
    ]

    for msg in conversation_messages:
        messages.append({
            "role": msg['role'],
            "content": msg['content']
        })

    return messages


def chat_with_history(user_message: str, conversation_messages: List[Dict]) -> str:
    """
    Conduct multiple rounds of dialogue based on historical messages
    
    parameters:
        user_message: User's current message
        conversation_messages: Historical message list
        
    Return:
        Assistant reply
    """
    try:
        # Build message list
        messages = get_chat_messages(conversation_messages, user_message)
        messages.append({"role": "user", "content": user_message})

        stream = get_deepseek_client().chat.completions.create(
            model="deepseek-v4-flash",
            messages=messages,
            stream=False
        )
        return stream.choices[0].message.content
    except Exception as e:
        logger.error(f"Multiple rounds of dialogue calling error: {e}")
        return "Sorry, I encountered some problems, please try again later."
