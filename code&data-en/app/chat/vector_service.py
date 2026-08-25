import numpy as np
import json
import logging
import os
from django.conf import settings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from sentence_transformers import SentenceTransformer
from .ai_service import get_deepseek_client, language_follow_instruction


logger = logging.getLogger(__name__)


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

def vector_similarity(que):
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
        "Reliable information was not found in the current corpus, so the system "
        "cannot generate an experimental recommendation. Please consult the relevant "
        "beamline staff for further assistance."
    )

    model = SentenceTransformer(settings.VECTOR_MODEL_NAME)

    with open(settings.VECTOR_K_EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
        embeddingsys1 = json.load(f)

    with open(settings.VECTOR_V_EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
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

    print("Highest query-to-key similarity:", best_key_score)

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
            "The following knowledge-base entry directly matches the user's question. "
            "Answer strictly from this entry. Do not introduce unsupported experimental "
            "parameters or information outside the supplied context. If the context does "
            "not cover part of the question, state that limitation clearly.\n\n"
            f"Strict context: {strict_context}"
        )

        print("Similarity >= 0.8; using strict context:")
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
            "The following entries are guidance context retrieved from the knowledge base. "
            "Answer using only information supported by these entries. Do not invent "
            "experimental conditions or parameters. Explain any differences among the "
            "retrieved entries.\n\n"
            f"Guidance context: {guidance_context}"
        )

        print("0.6 <= similarity < 0.8; using guidance context:")
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

        print("Highest query-to-value similarity:", best_value_score)
        print("Best matching answer content:", best_value_text)

        # -----------------------------------------------------
        # 4a. Answer similarity >= 0.4:
        #     Provide the retrieved answer as supporting context.
        # -----------------------------------------------------
        if best_value_score >= SUPPORT_THRESHOLD:
            prompt = (
                "The following text is supporting context retrieved from the knowledge "
                "base. Answer from this context and do not add unsupported experimental "
                "recommendations or parameters.\n\n"
                f"Supporting context: {best_value_text}"
            )

            print("Value similarity >= 0.4; using supporting context.")

            return wd_vector(que, prompt), best_value_score

        # -----------------------------------------------------
        # 4b. Answer similarity < 0.4:
        #     Do not call the LLM or generate a recommendation.
        # -----------------------------------------------------
        else:
            print("Value similarity < 0.4; no recommendation generated.")
            print(NO_RELIABLE_INFO_MESSAGE)

            return NO_RELIABLE_INFO_MESSAGE, best_value_score

# Backward-compatible alias for callers using the original function name.
vector_silimar = vector_similarity


if __name__ == "__main__":
    que = 'in4B7B-softXWhen performing absorption spectrum experiments at the ray experiment station, what should users pay attention to in terms of "turning on" and "turning off" operations and copying data after the experiment?'
    #que = 'I have a batch of film that I plan to1W1A-Diffuse scattering experimental stationGIWAXS Batch testing. Excuse me1W1A carry outGIWAXS When measuring, what are the recommended dimensions and substrate requirements for the sample (how to prepare the sample)?'
    #que = 'in1W1A-Diffuse scattering experimental station to make thin filmsGIWAXS What specific recommendations are there for sample preparation and batch testing preparation before experimentation?'
    #que = 'I'm1W1B-XAFS There is a batch of powder samples at the experimental station. What are the main steps in sample preparation?'
    #que = 'if in1W1B-XAFS The experimental station needs to measure liquid samples. What are the officially recommended steps for preparing liquid samples?'
    #que = 'in4W1B-XX-ray fluorescence microanalysis experimental stationXRF mapping How do samples usually need to be processed? Are there any recommended instructions for sample sectioning or tape use?'
    print(vector_similarity(que))
