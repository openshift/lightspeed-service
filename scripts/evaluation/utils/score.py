"""Score calculation for evaluation."""

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from rouge_score.rouge_scorer import RougeScorer
from scipy.spatial.distance import cosine, euclidean


class ResponseScore:
    """Calculate response score."""

    def __init__(self):
        """Initialize."""
        self._embedding_model = HuggingFaceEmbedding(
            "sentence-transformers/all-mpnet-base-v2"
        )
        self._rouge_scorer = RougeScorer(["rougeL"], use_stemmer=True)

    def calculate_scores(self, answer, response):
        """Calculate different similarity scores for two strings."""
        res_vec = self._embedding_model.get_text_embedding(response)
        ans_vec = self._embedding_model.get_text_embedding(answer)

        # Distance score
        cos_score = 1 - cosine(res_vec, ans_vec)
        euc_score = 1 - euclidean(res_vec, ans_vec)

        len_res, len_ans = len(response), len(answer)
        len_score = 1 - (abs(len_res - len_ans) / (len_res + len_ans))

        # text based scores
        rouge_score = self._rouge_scorer.score(target=answer, prediction=response)

        print(
            f"cos_score: {cos_score}, "
            f"euc_score: {euc_score}, "
            f"len_score: {len_score}, "
            f"rouge_score: {rouge_score}"
        )
        return (
            cos_score,
            euc_score,
            len_score,
            rouge_score["rougeL"].precision,
            rouge_score["rougeL"].recall,
            rouge_score["rougeL"].fmeasure,
        )
