"""Response evaluation using pre-defined question/answer pair."""

import json
import os
from collections import defaultdict
from datetime import UTC, datetime
from time import sleep

from pandas import DataFrame, concat, read_csv, read_parquet
from tqdm import tqdm

from ols import config

from .utils.constants import (
    DEFAULT_CONFIG_FILE,
    DEFAULT_INPUT_DIR,
    DEFAULT_QNA_FILE,
    DEFAULT_RESULT_DIR,
    EVAL_MODES,
    EVAL_THRESHOLD,
    INSCOPE_MODELS,
    LLM_BASED_EVALS,
    MAX_RETRY_ATTEMPTS,
    SCORE_DESCRIPTION,
    TIME_TO_BREATH,
)
from .utils.plot import plot_score
from .utils.response import get_model_response
from .utils.score import ResponseScore

tqdm.pandas()


# TODO: OLS-712 Enrichment of Q+A pairs to contain questions with attachments
class ResponseEvaluation:
    """Evaluate LLM response."""

    def __init__(self, eval_args, api_client):
        """Initialize."""
        print(f"Response evaluation arguments: {eval_args}")
        self._args = eval_args
        self._api_client = api_client

        self._validate_args()
        self._load_config_and_rag()  # Set global config
        self._input_dir, self._result_dir = self._set_directories()

        self._scorer = ResponseScore(self._args)

        # Load data
        with open(
            os.path.join(self._input_dir, DEFAULT_QNA_FILE), encoding="utf-8"
        ) as qna_f:
            self._qa_pool_json = json.load(qna_f)["evaluation"]

        self._qa_pool_df = self._load_qna_pool_parquet()

    def _validate_args(self):
        """Validate key arguments."""
        invalid_provider_model = set(self._args.eval_provider_model_id) - set(
            INSCOPE_MODELS.keys()
        )
        if len(invalid_provider_model) > 0:
            raise ValueError(f"Invalid Provider/Model IDs: {invalid_provider_model}")
        invalid_metrics = set(self._args.eval_metrics) - set(SCORE_DESCRIPTION.keys())
        if len(invalid_metrics) > 0:
            raise ValueError(f"Invalid metrics: {invalid_metrics}")
        invalid_modes = set(self._args.eval_modes) - EVAL_MODES
        if len(invalid_modes) > 0:
            raise ValueError(f"Invalid eval modes: {invalid_modes}")

    def _load_config_and_rag(self):
        """Load config and RAG."""
        if (len(set(self._args.eval_modes) - {"ols"}) > 0) or (
            len(set(self._args.eval_metrics).intersection(set(LLM_BASED_EVALS.keys())))
            > 0
        ):
            # load config separately
            # Use OLS config file to set provider/model related config. Ex: credential/url
            cfg_file = os.environ.get("OLS_CONFIG_FILE", DEFAULT_CONFIG_FILE)
            config.reload_from_yaml_file(cfg_file)

        if "ols_rag" in self._args.eval_modes:
            # load rag index
            config.rag_index  # pylint: disable=W0104
            if config.rag_index is None:
                raise Exception("No valid rag index for ols_rag mode")

    def _set_directories(self):
        """Set input/output directories."""
        eval_dir = os.path.dirname(__file__)
        input_dir = os.path.join(eval_dir, DEFAULT_INPUT_DIR)

        result_dir = os.path.join(
            (self._args.eval_out_dir or eval_dir), DEFAULT_RESULT_DIR
        )
        os.makedirs(result_dir, exist_ok=True)
        return input_dir, result_dir

    def _load_qna_pool_parquet(self):
        """Load QnA pool from parquet file."""
        qna_pool_df = DataFrame()
        if self._args.qna_pool_file is not None:
            qna_pool_df = read_parquet(self._args.qna_pool_file).reset_index()
            qna_pool_df.drop_duplicates(inplace=True)
            qna_pool_df = qna_pool_df.rename(
                columns={"ID": "query_id", "Question": "question", "Answer": "answer"}
            )
            qna_pool_df["query_id"] = "qna" + qna_pool_df["query_id"].astype(str)
            qna_pool_df["query_source"] = "doc"
            qna_pool_df["consistency_cutoff"] = EVAL_THRESHOLD
            qna_pool_df["in_use"] = True
        return qna_pool_df

    def _restructure_qna_pool_json(self, provider_model_id):
        """Restructure qna pool json data to dataframe."""
        qna_pool_dict = defaultdict(list)

        provider, model = INSCOPE_MODELS[provider_model_id]
        answer_id = f"{provider}+{model}+{self._args.eval_scenario}"
        cutoff_score_id = None

        if self._args.eval_type == "model":
            cutoff_score_id = answer_id
            answer_id = f"ground_truth+{self._args.eval_scenario}"

        for query_id in self._qa_pool_json.keys():
            question = self._qa_pool_json[query_id]["question"]
            answer_data = self._qa_pool_json[query_id]["answer"][answer_id]
            consistency_cutoff = answer_data.get("cutoff_score", EVAL_THRESHOLD)
            if isinstance(consistency_cutoff, dict):
                consistency_cutoff = consistency_cutoff.get(
                    cutoff_score_id, EVAL_THRESHOLD
                )
            in_use = answer_data.get("in_use", True)

            for answer in answer_data["text"]:
                qna_pool_dict["query_id"].append(query_id)
                qna_pool_dict["question"].append(question)
                qna_pool_dict["answer"].append(answer)
                qna_pool_dict["query_source"].append("transcript")
                qna_pool_dict["doc_source"].append("NA")
                qna_pool_dict["doc_title"].append("NA")
                qna_pool_dict["doc_page"].append("NA")
                qna_pool_dict["consistency_cutoff"].append(consistency_cutoff)
                qna_pool_dict["in_use"].append(in_use)

        return DataFrame.from_dict(qna_pool_dict)

    def _get_inscope_qna(self, provider_model_id):
        """Get QnAs which are inscope for evaluation."""
        qna_pool_df = self._restructure_qna_pool_json(provider_model_id)

        qna_pool_df = concat([qna_pool_df, self._qa_pool_df])

        if self._args.eval_query_ids is not None:
            qna_pool_df = qna_pool_df[
                qna_pool_df.query_id.isin(self._args.eval_query_ids)
            ]
        qna_pool_df = qna_pool_df[qna_pool_df.in_use]
        return qna_pool_df.reset_index(drop=True).drop(columns="in_use")

    def _get_api_response(
        self,
        question,
        provider,
        model,
        eval_mode,
        retry_attempts=MAX_RETRY_ATTEMPTS,
        time_to_breath=TIME_TO_BREATH,
    ):
        """Get api response for a question/query."""
        response = None
        # try to retrieve response even when model is not responding reliably
        # in 100% cases
        # it fixes OLS-858 e2e test failure - test_model_response response validation
        for retry_counter in range(retry_attempts):
            print(f"Call attempt: {retry_counter}")
            try:
                response = get_model_response(
                    question,
                    provider,
                    model,
                    eval_mode,
                    self._api_client,
                )
                break
            except Exception:
                if retry_counter == retry_attempts - 1:
                    raise
            # model is not realiable if it's overloaded, so take some time between requests
            sleep(time_to_breath)

        print(
            f"API request is successful for {provider}+{model}; "
            f"mode: {eval_mode};\nQuery: {question}"
        )
        return response

    def _get_recent_response(
        self, question, recent_resp_df, provider, model, eval_mode
    ):
        """Get llm response from the stored data, if available."""
        if recent_resp_df is not None:
            try:
                return recent_resp_df[recent_resp_df.question == question][
                    "response"
                ].iloc[0]
            except IndexError:
                print(
                    "Recent response for query is not found in the file. "
                    "Separate api call is required to get response."
                )
        # Recent response is not found, call api to get response
        return self._get_api_response(question, provider, model, eval_mode)

    def _get_model_response(self, qna_pool_df, provider_model_id, eval_mode):
        """Get model responses for all questions."""
        temp_resp_file = (
            f"{self._result_dir}/{eval_mode}_"
            f"{provider_model_id.replace('/', '-')}_response.csv"
        )
        try:
            recent_resp_df = read_csv(temp_resp_file)
        except FileNotFoundError:
            print(
                "File with recent model response not found. "
                "Separate api calls are required to get the response."
            )
            recent_resp_df = None

        provider, model = INSCOPE_MODELS[provider_model_id]
        qna_pool_unique = qna_pool_df[["question"]].drop_duplicates()
        qna_pool_unique["response"] = qna_pool_unique.progress_apply(
            lambda row: self._get_recent_response(
                row.question, recent_resp_df, provider, model, eval_mode
            ),
            axis=1,
        )
        qna_pool_df = qna_pool_df.merge(qna_pool_unique, on="question")

        qna_pool_df.to_csv(temp_resp_file, index=False)
        return qna_pool_df

    def _get_evaluation_score(self, qna_pool_df):
        """Get response evaluation score."""
        print("Getting evaluation scores...")
        # Default scores
        score_cols = [
            "cos_score",
            "euc_score",
            "len_score",
            "rougeL_precision",
            "rougeL_recall",
            "rougeL_f1",
            "answer_relevancy",
            "answer_valid_flag",  # Supporting data for answer_relevancy
            "generated_questions",  # Supporting data for answer_relevancy
            "answer_similarity_llm",
        ]
        qna_pool_df[score_cols] = qna_pool_df.progress_apply(
            lambda row: self._scorer.calculate_scores(
                row.question, row.answer, row.response
            ),
            axis=1,
            result_type="expand",
        )
        return qna_pool_df.dropna(axis=1, how="all")

    def _get_response_with_score(self):
        """Get responses with scores."""
        result_dfs = []
        for provider_model_id in self._args.eval_provider_model_id:
            for eval_mode in self._args.eval_modes:
                print(f"Model evaluation for {provider_model_id}; Mode: {eval_mode}")
                try:
                    qna_pool_df = read_csv(
                        f"{self._result_dir}/temp_score-{eval_mode}-"
                        f"{provider_model_id.replace('/', '-')}.csv"
                    )
                    print("Temp score file exists. Proceeding without calculation.")
                except Exception:
                    print("Temp score doesn't exist. Proceeding with calculation.")
                    qna_pool_df = self._get_inscope_qna(provider_model_id)
                    qna_pool_df = self._get_model_response(
                        qna_pool_df, provider_model_id, eval_mode
                    )
                    qna_pool_df = self._get_evaluation_score(qna_pool_df)
                    qna_pool_df["eval_mode"] = eval_mode
                    qna_pool_df["provider_model_id"] = provider_model_id
                    qna_pool_df.to_csv(
                        f"{self._result_dir}/temp_score-{eval_mode}-"
                        f"{provider_model_id.replace('/', '-')}.csv",
                        index=False,
                    )
                result_dfs.append(qna_pool_df)
        return concat(result_dfs)

    @staticmethod
    def _condense_eval_df(result_df):
        """Put all models' result as columns."""
        result_df = result_df.pivot(
            index=[
                "query_id",
                "question",
                "answer",
                "query_source",
                "doc_source",
                "doc_title",
                "doc_page",
            ],
            columns=["eval_mode", "provider_model_id"],
        ).swaplevel(0, axis=1)
        result_df.columns = ["_".join(col) for col in result_df.columns]
        return result_df

    def validate_response(self):
        """Validate LLM response."""
        consistency_success_flag = True
        result_df = self._get_response_with_score()

        # TODO: Other scores have been changed to make higher value better.
        # But keeping consistency score same as earlier, as it requires cut-off adjustment.
        result_df["consistency_score"] = (1 - result_df.len_score) * 0.1 + (
            (1 - result_df.cos_score) + (1 - result_df.euc_score)
        ) / 2

        if len(result_df) > 0:
            result_df["answer_eval_fail_flag"] = (
                result_df.consistency_score > result_df.consistency_cutoff
            )
            # If none of the answer for any question has score below threshold,
            # then mark as evaluation failure.
            result_df["question_eval_fail_flag"] = result_df.groupby(
                "query_id"
            ).answer_eval_fail_flag.transform("min")

            # If evaluation has failed for any question,
            # then return False (Failed validation scenario)
            consistency_success_flag = result_df.question_eval_fail_flag.max() == 0

            result_df = self._condense_eval_df(result_df)
            result_file = (
                f"{self._result_dir}/response_evaluation_result-"
                f"{':'.join(self._args.eval_provider_model_id).replace('/', '-')}+with_rag.csv"
            )
            result_df.to_csv(result_file)
            print(f"Result is saved to {result_file}")
            if not consistency_success_flag:
                print(
                    "Response is not matching for question(s): "
                    "Please check result file."
                )
        else:
            print("No result. Nothing to process.")

        return consistency_success_flag

    def evaluate_models(self):
        """Evaluate models against groundtruth answer."""
        print("Running model evaluation using groundtruth...")
        result_df = self._get_response_with_score()
        result_df = self._condense_eval_df(result_df)
        result_df.to_csv(f"{self._result_dir}/model_evaluation_result.csv")

        result_df = result_df.groupby(level="query_id").max(numeric_only=True)

        summary_score = {}
        for score_type in self._args.eval_metrics:
            num_columns = [
                f"{pm}_{m}_{score_type}"
                for pm in self._args.eval_provider_model_id
                for m in self._args.eval_modes
            ]
            temp_result_df = result_df[num_columns]
            temp_result_df.columns = [
                col.removesuffix(f"_{score_type}") for col in temp_result_df.columns
            ]
            plot_file = f"{self._result_dir}/model_evaluation_result-{score_type}.png"
            plot_score(temp_result_df, SCORE_DESCRIPTION[score_type], plot_file)
            summary_score[score_type] = temp_result_df.describe().T.to_dict()

        summary_result = {
            "timestamp": str(datetime.now(UTC)),
            "score": summary_score,
        }
        # Save model evaluation summary report
        with open(
            f"{self._result_dir}/model_evaluation_summary.json", "w", encoding="utf-8"
        ) as f:
            json.dump(summary_result, f)
