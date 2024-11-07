"""Relevancy score calculation."""

from statistics import mean
from time import sleep

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from scipy.spatial.distance import cosine

from ols import config

from .constants import MAX_RETRY_ATTEMPTS, N_QUESTIONS, TIME_TO_BREATH
from .models import VANILLA_MODEL
from .prompts import ANSWER_RELEVANCY_PROMPT


class AnswerRelevancyScore:
    """Calculate response/answer relevancy score."""

    def __init__(
        self, embedding_model, judge_provider="ollama", judge_model="llama3.1:latest"
    ):
        """Initialize."""
        self._embedding_model = embedding_model
        self._judge_llm = self._judge_llm_init(judge_provider, judge_model)

    @staticmethod
    def _judge_llm_init(judge_provider, judge_model):
        """Load judge LLM."""
        # Provider/model should be in config yaml
        provider_config = config.config.llm_providers.providers[judge_provider]
        llm = VANILLA_MODEL[provider_config.type](judge_model, provider_config).load()

        prompt = PromptTemplate.from_template(ANSWER_RELEVANCY_PROMPT)
        return prompt | llm | JsonOutputParser()

    def get_score(
        self,
        question,
        response,
        retry_attemps=MAX_RETRY_ATTEMPTS,
        time_to_breath=TIME_TO_BREATH,
    ):
        """Calculate relevancy score."""
        # Generate relevant questions.
        for retry_counter in range(retry_attemps):
            try:
                out = self._judge_llm.invoke(
                    {"answer": response, "num_questions": N_QUESTIONS}
                )
                break
            except Exception:
                if retry_counter == retry_attemps - 1:
                    out = None  ## Continue with without result
                    # raise
            sleep(time_to_breath)

        if out:
            valid_flag = out["Valid"]
            gen_questions = out["Question"]
            score = 0
            if valid_flag == 1:
                org_vec = self._embedding_model.get_text_embedding(question)
                score = mean(
                    [
                        1
                        - cosine(
                            org_vec,
                            self._embedding_model.get_text_embedding(gen_question),
                        )
                        for gen_question in gen_questions
                    ]
                )

            return score, valid_flag, "\n".join(gen_questions)

        return None, None, None
