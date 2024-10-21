"""Relevancy score calculation."""

from time import sleep

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy

from .utils.constants import MAX_RETRY_ATTEMPTS, TIME_TO_BREATH


class AnswerRelevancyScore:
    """Calculate response/answer relevancy score."""

    def __init__(self):
        """Initialize."""
        self._embedding_model_init()
        # Currently using ollama for LLM. RHELAI/RHOAI can also be used.
        self._local_ollama_llm_init()
        self._scorer_init()

    def _embedding_model_init(self):
        """Set up embedding model."""
        embed_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        self._embeddings = LangchainEmbeddingsWrapper(embed_model)

    def _local_ollama_llm_init(self):
        """Set up local ollama inference."""
        # Ollama must be running.
        base_llm = ChatOpenAI(
            base_url="http://127.0.0.1:11434/v1",  # ollama url
            # model="mistral",
            model="llama3.1:latest",
            openai_api_key="IGNORED",
        )
        self._llm = LangchainLLMWrapper(base_llm)

    def _scorer_init(self):
        """Initialize scorer."""
        self._scorer = AnswerRelevancy(
            llm=self._llm,
            embeddings=self._embeddings,
        )

    def get_score(
        self,
        question,
        response,
        retry_attemps=MAX_RETRY_ATTEMPTS,
        time_to_breath=TIME_TO_BREATH,
    ):
        """Calculate relevancy score."""
        for retry_counter in range(retry_attemps):
            try:
                eval_data = SingleTurnSample(
                    user_input=question,
                    response=response,
                )
                score = self._scorer.single_turn_score(eval_data)
                break
            except Exception:
                if retry_counter == retry_attemps - 1:
                    score = None  ## Continue with score as None
                    # raise
            sleep(time_to_breath)
        return score
