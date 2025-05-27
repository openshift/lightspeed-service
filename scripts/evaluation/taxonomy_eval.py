"""Taxonomy Answer/Context Evaluation."""

import argparse
import os
import sys
from time import sleep

import yaml
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from pandas import DataFrame
from tqdm import tqdm

from ols import config

from .utils.constants import (
    DEFAULT_CONFIG_FILE,
    DEFAULT_RESULT_DIR,
    MAX_RETRY_ATTEMPTS,
    TIME_TO_BREATH,
)
from .utils.models import VANILLA_MODEL
from .utils.prompts import GROUNDNESS_PROMPT, TAXONOMY_CONTEXT_RELEVANCY

tqdm.pandas()


# Sample taxonomy file
# https://github.com/instructlab/taxonomy/blob/main/knowledge/arts/music/fandom/swifties/qna.yaml


def _args_parser(args):
    """Arguments parser."""
    parser = argparse.ArgumentParser(description="RAG evaluation module.")
    parser.add_argument(
        "--judge_provider",
        default="watsonx",
        type=str,
        help="Provider name for judge model; required for LLM based evaluation",
    )
    parser.add_argument(
        "--judge_model",
        default="meta-llama/llama-3-1-8b-instruct",
        type=str,
        help="Judge model; required for LLM based evaluation",
    )
    parser.add_argument(
        "--taxonomy_file_path",
        type=str,
        help="Taxonomy file path.",
    )
    parser.add_argument(
        "--eval_out_dir",
        default=None,
        type=str,
        help="Result destination.",
    )
    parser.add_argument(
        "--eval_type",
        choices=["context", "answer", "all"],
        default="answer",
        type=str,
        help="Evaluation type",
    )
    parser.add_argument(
        "--eval_method",
        choices=["ragas", "custom"],
        default="custom",
        type=str,
        help="Evaluation type",
    )
    return parser.parse_args(args)


class TaxonomyEval:
    """Evaluate taxonomy answer/context."""

    def __init__(self, eval_args):
        """Initialize."""
        print(f"Arguments: {eval_args}")
        self._args = eval_args

        self._load_judge()  # Set global config
        self._set_output_dir()
        self._load_taxonomy_yaml()

    def _load_judge(self):
        """Load Judge."""
        print("Loading judge model...")
        # Load config separately
        # Use OLS config file to set Judge provider/model
        cfg_file = os.environ.get("OLS_CONFIG_FILE", DEFAULT_CONFIG_FILE)
        config.reload_from_yaml_file(cfg_file)

        provider_config = config.config.llm_providers.providers[
            self._args.judge_provider
        ]
        assert provider_config.type is not None, "Provider type must be configured"
        self._judge_llm = VANILLA_MODEL[provider_config.type](
            self._args.judge_model, provider_config
        ).load()

    def _set_output_dir(self):
        """Set output directory."""
        eval_dir = os.path.dirname(__file__)

        result_dir = os.path.join(
            (self._args.eval_out_dir or eval_dir), DEFAULT_RESULT_DIR
        )
        os.makedirs(result_dir, exist_ok=True)
        self._result_dir = result_dir

    def _load_taxonomy_yaml(self):
        """Load taxonomy YAML file."""
        print("Loading taxonomy file...")
        with open(self._args.taxonomy_file_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        data_f = [
            {**qna, "context": ex["context"]}
            for ex in data["seed_examples"]
            for qna in ex["questions_and_answers"]
        ]
        self._taxonomy_df = DataFrame(data_f)

    def _get_judge_response(self, question, answer, context, prompt):
        """Get Judge response."""
        print("Getting Judge response...")
        result = None
        prompt = PromptTemplate.from_template(prompt)
        judge_llm = prompt | self._judge_llm | JsonOutputParser()

        for retry_counter in range(MAX_RETRY_ATTEMPTS):
            try:
                result = judge_llm.invoke(
                    {
                        "question": question,
                        "answer": answer,
                        "context": context,
                    }
                )
                break
            except Exception as e:
                if retry_counter == MAX_RETRY_ATTEMPTS - 1:
                    print(f"error_groundness_score: {e}")
                    # Continue with empty result
                    result = {}
            sleep(TIME_TO_BREATH)

        return result

    def _get_score(self, df, scores, prompt):
        """Get score."""
        df["score"] = df.progress_apply(
            lambda row: self._get_judge_response(
                row.question, row.answer, row.context, prompt
            ),
            axis=1,
            # result_type="expand",
        )
        for s in scores:
            df[s] = df["score"].apply(lambda x: x.get(s, None))  # pylint: disable=W0640
        return df

    def _get_custom_score(self):
        """Get custom score."""
        df = self._taxonomy_df.copy()
        if self._args.eval_type in ("all", "context"):
            scores = ["valid_flag", "relevancy_score"]
            df = self._get_score(df, scores, TAXONOMY_CONTEXT_RELEVANCY)
            renamed_columns = {score: f"context_{score}" for score in scores}
            df.rename(columns=renamed_columns, inplace=True)
        if self._args.eval_type in ("all", "answer"):
            scores = ["relevancy_score", "groundness_score"]
            df = self._get_score(df, scores, GROUNDNESS_PROMPT)
            renamed_columns = {score: f"answer_{score}" for score in scores}
            df.rename(columns=renamed_columns, inplace=True)
        df.drop(columns=["score"], inplace=True)
        return df

    def _get_ragas_score(self):
        """Get ragas score."""
        # pylint: disable=C0415
        from ragas import SingleTurnSample
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference

        judge_llm = LangchainLLMWrapper(self._judge_llm)

        def _get_score(data, scorer):
            data = SingleTurnSample(
                user_input=data.question,
                response=data.answer,
                retrieved_contexts=[data.context],
            )
            return scorer.single_turn_score(data)

        df = self._taxonomy_df.copy()
        if self._args.eval_type in ("all", "context"):
            scorer = LLMContextPrecisionWithoutReference(llm=judge_llm)
            df["context_relevancy_score"] = df.progress_apply(
                lambda x: _get_score(x, scorer), axis=1
            )
        if self._args.eval_type in ("all", "answer"):
            scorer = Faithfulness(llm=judge_llm)
            df["answer_groundness_score"] = df.progress_apply(
                lambda x: _get_score(x, scorer), axis=1
            )
        return df

    def get_eval_result(self):
        """Get evaluation result."""
        if self._args.eval_method == "ragas":
            result_df = self._get_ragas_score()
        else:
            result_df = self._get_custom_score()

        result_df.to_csv(
            f"{self._result_dir}/context_eval-{self._args.eval_method}.csv", index=False
        )


def main():
    """Evaluate taxonomy context/answer."""
    args = _args_parser(sys.argv[1:])
    TaxonomyEval(args).get_eval_result()


if __name__ == "__main__":
    main()
