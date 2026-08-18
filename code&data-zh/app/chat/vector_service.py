import numpy as np
import json
from openai import OpenAI
import pandas as pd
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from sentence_transformers import SentenceTransformer
from .ai_service import language_follow_instruction


client = OpenAI(api_key="yours", base_url="https://api.deepseek.com")
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

def vector_silimar(que):
    """
    Similarity thresholds were manually selected during system development.
    They are specific to the embedding model and corpus used in this study
    and should not be regarded as generally calibrated values.
    """

    # Fixed thresholds selected manually based on development experience.
    STRICT_THRESHOLD = 0.8
    GUIDANCE_THRESHOLD = 0.6
    SUPPORT_THRESHOLD = 0.4

    NO_RELIABLE_INFO_MESSAGE = (
        "抱歉，当前语料库中没有找到足够可靠的信息，"
        "因此暂时无法生成实验建议。"
        "建议您咨询相关光束线工作人员，以获得进一步帮助。"
    )

    model = SentenceTransformer("all-MiniLM-L6-v2")

    with open("./data/k_embeddings.json", "r", encoding="utf-8") as f:
        embeddingsys1 = json.load(f)

    with open("./data/v_embeddings.json", "r", encoding="utf-8") as f:
        embeddingsys2 = json.load(f)

    k_sentences = embeddingsys1["sentences"]
    k_embeddings = embeddingsys1["embeddings"]

    v_sentences = embeddingsys2["sentences"]
    v_embeddings = embeddingsys2["embeddings"]

    # Encode the user query.
    txt_emb = model.encode(que)
    txt_emb2 = np.array(txt_emb, dtype=np.float32).reshape(1, -1)

    # ---------------------------------------------------------
    # 1. Calculate similarity between the query and knowledge keys.
    # ---------------------------------------------------------
    k_embeddings2 = np.array(k_embeddings, dtype=np.float32)
    similarity_matrix = model.similarity(txt_emb2, k_embeddings2)
    similarities = similarity_matrix[0].tolist()

    score_index_sentence = [
        (float(similarities[i]), i, k_sentences[i])
        for i in range(len(k_sentences))
    ]

    score_index_sentence.sort(
        key=lambda x: x[0],
        reverse=True
    )

    best_key_score, best_key_index, best_key_sentence = (
        score_index_sentence[0]
    )

    print("用户问题与知识库问题的最高相似度得分：", best_key_score)

    # ---------------------------------------------------------
    # 2. Similarity >= 0.8:
    #    Use the matched question-answer pair as strict context.
    # ---------------------------------------------------------
    if best_key_score >= STRICT_THRESHOLD:
        matched_answer = v_sentences[best_key_index]

        strict_context = [
            (best_key_sentence, matched_answer)
        ]

        prompt = (
            "以下是与用户问题直接匹配的知识库问答内容。"
            "请严格依据该内容回答用户问题，不要引入知识库之外的"
            "实验参数或未经证实的信息。"
            "如果知识库内容没有覆盖用户问题，请明确说明。\n\n"
            f"严格参考内容：{strict_context}"
        )

        print("相似度 >= 0.8，使用严格上下文：")
        print(strict_context)

        return wd_vector(que, prompt), best_key_score

    # ---------------------------------------------------------
    # 3. 0.6 <= Similarity < 0.8:
    #    Use the retrieved question-answer pairs as guidance context.
    # ---------------------------------------------------------
    elif best_key_score >= GUIDANCE_THRESHOLD:
        top5_results = score_index_sentence[:5]

        guidance_context = [
            (
                k_sentences[index],
                v_sentences[index]
            )
            for _, index, _ in top5_results
        ]

        prompt = (
            "以下内容是从知识库中检索到的指导性上下文。"
            "请结合这些内容回答用户问题，只使用上下文支持的"
            "信息，不要擅自编造实验条件或参数。"
            "如果不同内容之间存在差异，请谨慎说明。\n\n"
            f"指导性上下文：{guidance_context}"
        )

        print("0.6 <= 相似度 < 0.8，使用指导性上下文：")
        print(guidance_context)

        return wd_vector(que, prompt), best_key_score

    # ---------------------------------------------------------
    # 4. Similarity < 0.6:
    #    Calculate similarity between the query and answer contents.
    # ---------------------------------------------------------
    else:
        v_embeddings2 = np.array(v_embeddings, dtype=np.float32)

        similarity_matrix2 = model.similarity(
            txt_emb2,
            v_embeddings2
        )

        # Important:
        # Use similarity_matrix2 here, not similarity_matrix.
        value_similarities = similarity_matrix2[0].tolist()

        score_index_sentence2 = [
            (
                float(value_similarities[i]),
                i,
                v_sentences[i]
            )
            for i in range(len(v_sentences))
        ]

        best_value_score, best_value_index, best_value_text = max(
            score_index_sentence2,
            key=lambda x: x[0]
        )

        print("问题与知识库答案内容的最高相似度得分：", best_value_score)
        print("最匹配的答案内容：", best_value_text)

        # -----------------------------------------------------
        # 4a. Answer similarity >= 0.4:
        #     Provide the retrieved answer as supporting context.
        # -----------------------------------------------------
        if best_value_score >= SUPPORT_THRESHOLD:
            prompt = (
                "以下内容是从知识库中检索到的支持性上下文。"
                "请将其作为辅助信息回答用户问题。"
                "回答必须以该内容为依据，不要添加未经知识库支持的"
                "实验建议或参数。\n\n"
                f"支持性上下文：{best_value_text}"
            )

            print("答案内容相似度 >= 0.4，使用支持性上下文。")

            return wd_vector(que, prompt), best_value_score

        # -----------------------------------------------------
        # 4b. Answer similarity < 0.4:
        #     Do not call the LLM or generate a recommendation.
        # -----------------------------------------------------
        else:
            print("答案内容相似度 < 0.4，不生成实验建议。")
            print(NO_RELIABLE_INFO_MESSAGE)

            return NO_RELIABLE_INFO_MESSAGE, best_value_score

if __name__ == "__main__":
    que = '在 4B7B-软X射线实验站做吸收谱实验时，用户在“开光”和“关光”操作以及实验结束后的数据拷贝方面需要注意什么？'
    #que = '我有一批薄膜，计划在 1W1A-漫散射实验站做 GIWAXS 批量测试。请问在 1W1A 进行 GIWAXS 测量时，样品的推荐尺寸和基底要求是什么（样品怎么准备）？'
    #que = '在 1W1A-漫散射实验站做薄膜 GIWAXS 实验前，样品制备和批量测试准备有哪些具体建议？'
    #que = '我在 1W1B-XAFS 实验站有一批粉末样品，样品制备的主要步骤是什么？'
    #que = '如果在 1W1B-XAFS 实验站要测量液体样品，官方推荐的液体样品制备步骤大致是什么？'
    #que = '在 4W1B-X射线荧光微分析实验站做 XRF mapping 时，样品通常需要如何处理？是否有推荐的样品切片或胶带使用说明？'
    print(vector_silimar(que))
