"""Score calculation for evaluation."""

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from rouge_score.rouge_scorer import RougeScorer
from scipy.spatial.distance import cosine, euclidean

from ols import config

from .constants import LLM_BASED_EVALS
from .models import VANILLA_MODEL
from .relevancy_score import AnswerRelevancyScore
from .similarity_score_llm import AnswerSimilarityScore


class ResponseScore:
    """Calculate response score."""

    def __init__(self, args):
        """Initialize."""
        self._embedding_model = HuggingFaceEmbedding(
            "sentence-transformers/all-mpnet-base-v2"
        )

        self._rouge_scorer = RougeScorer(["rougeL"], use_stemmer=True)
        self._relevancy_scorer = None
        self._llm_similarity_scorer = None

        judge_llm_required = set(args.eval_metrics).intersection(
            set(LLM_BASED_EVALS.keys())
        )
        if judge_llm_required:
            # Judge provider & model need to be configured correctly in config yaml file.
            provider_config = config.config.llm_providers.providers[args.judge_provider]
            assert provider_config.type is not None, "Provider type must be configured"
            judge_llm = VANILLA_MODEL[provider_config.type](
                args.judge_model, provider_config
            ).load()
            if "answer_relevancy" in judge_llm_required:
                self._relevancy_scorer = AnswerRelevancyScore(
                    judge_llm, self._embedding_model
                )
            if "answer_similarity_llm" in judge_llm_required:
                self._llm_similarity_scorer = AnswerSimilarityScore(judge_llm)

    def calculate_scores(self, query, answer, response):
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

        relevancy_score = answer_valid_flag = generated_questions = None
        if self._relevancy_scorer:
            relevancy_score, answer_valid_flag, generated_questions = (
                self._relevancy_scorer.get_score(query, response)
            )
        llm_similarity_score = None
        if self._llm_similarity_scorer:
            llm_similarity_score = self._llm_similarity_scorer.get_score(
                query, answer, response
            )

        print(
            f"cos_score: {cos_score}, "
            f"euc_score: {euc_score}, "
            f"len_score: {len_score}, "
            f"rouge_score: {rouge_score}, "
            f"relevancy_score: {relevancy_score}, "
            f"llm_similarity_score: {llm_similarity_score}"
        )
        return (
            cos_score,
            euc_score,
            len_score,
            rouge_score["rougeL"].precision,
            rouge_score["rougeL"].recall,
            rouge_score["rougeL"].fmeasure,
            relevancy_score,
            # Return additional information
            answer_valid_flag,
            generated_questions,
            llm_similarity_score,
        )
