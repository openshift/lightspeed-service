"""Response evaluation using pre-defined question/answer pair."""

import json
import os
from collections import defaultdict
from datetime import UTC, datetime
from time import sleep

import matplotlib.pyplot as plt
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from matplotlib.colors import BASE_COLORS
from pandas import DataFrame, concat, read_csv, read_parquet
from rouge_score.rouge_scorer import RougeScorer
from scipy.spatial.distance import cosine, euclidean
from tqdm import tqdm

from ols import config

from .utils import get_model_response

tqdm.pandas()


# Same Provider/Model combination must be used while launching OLS.
INSCOPE_MODELS = {
    "bam+ibm/granite-13b-chat-v2": ("bam", "ibm/granite-13b-chat-v2"),
    "watsonx+ibm/granite-13b-chat-v2": ("watsonx", "ibm/granite-13b-chat-v2"),
    "openai+gpt-3.5-turbo": ("openai", "gpt-3.5-turbo"),
    "openai+gpt-4o-mini": ("openai", "gpt-4o-mini"),
    "azure_openai+gpt-3.5-turbo": ("azure_openai", "gpt-3.5-turbo"),
    "azure_openai+gpt-3.5-turbo-4k": ("azure_openai", "gpt-3.5-turbo"),
    "azure_openai+gpt-3.5-turbo-16k": ("azure_openai", "gpt-3.5-turbo"),
    "azure_openai+gpt-4o": ("azure_openai", "gpt-4o"),
}
SCORE_DESCRIPTION = {
    "cos_score": "Cosine Similarity Score (mpnet)",
    "euc_score": "Euclidean Distance Score (mpnet)",
    "len_score": "Character length delta Score",
    "rougeL_precision": "RougeL Precision Score",
    "rougeL_recall": "RougeL Recall Score",
    "rougeL_f1": "RougeL F1 Score",
}
EVAL_MODES = {
    "vanilla",
    "ols_param",
    "ols_prompt",
    "ols_rag",
    "ols",
}

DEFAULT_QNA_FILE = "question_answer_pair.json"

# Cut-off similarity score used for response evaluation.
EVAL_THRESHOLD = 0.2  # low score is better

# Retry settings for LLM calls used when model does not respond reliably in 100% cases
MAX_RETRY_ATTEMPTS = 10
REST_API_TIMEOUT = 120
TIME_TO_BREATH = 10


# TODO: OLS-712 Enrichment of Q+A pairs to contain questions with attachments
class ResponseEvaluation:
    """Evaluate LLM response."""

    def __init__(self, eval_args, api_client):
        """Initialize."""
        print(f"Response evaluation arguments: {eval_args}")
        self._args = eval_args
        self._api_client = api_client

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

        if len(set(self._args.eval_modes) - {"ols"}) > 0:
            # load config separately
            # Use OLS config file to set provider/model related config. Ex: credential/url
            cfg_file = os.environ.get("OLS_CONFIG_FILE", "olsconfig.yaml")
            config.reload_from_yaml_file(cfg_file)
        if "ols_rag" in self._args.eval_modes:
            # load rag index
            config.rag_index
            if config.rag_index is None:
                raise Exception("No valid rag index for ols_rag mode")

        self._embedding_model = HuggingFaceEmbedding(
            "sentence-transformers/all-mpnet-base-v2"
        )
        self._rouge_scorer = RougeScorer(["rougeL"], use_stemmer=True)

        eval_dir = os.path.dirname(__file__)
        self._input_dir = os.path.join(eval_dir, "eval_data")
        with open(os.path.join(self._input_dir, DEFAULT_QNA_FILE)) as qna_f:
            self._qa_pool_json = json.load(qna_f)["evaluation"]

        self._qa_pool_df = self._load_qna_pool_parquet()

        self._result_dir = os.path.join(
            (self._args.eval_out_dir or eval_dir), "eval_result"
        )
        os.makedirs(self._result_dir, exist_ok=True)

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
            qna_pool_df["consistency_cutoff"] = EVAL_THRESHOLD
            qna_pool_df["in_use"] = True
        return qna_pool_df

    def _restructure_qna_pool_json(self, provider_model_id):
        """Restructure qna pool json data to dataframe."""
        qna_pool_dict = defaultdict(list)

        provider, model = INSCOPE_MODELS[provider_model_id]
        answer_id = f"{provider}+{model}+{self._args.eval_scenario}"
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
                qna_pool_dict["doc_page"].append(None)
                qna_pool_dict["doc_title"].append(None)
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
        retry_attemps=MAX_RETRY_ATTEMPTS,
        rest_api_timeout=REST_API_TIMEOUT,
        time_to_breath=TIME_TO_BREATH,
    ):
        """Get api response for a question/query."""
        # try to retrieve response even when model is not responding reliably
        # in 100% cases
        # it fixes OLS-858 e2e test failure - test_model_response response validation
        for retry_counter in range(retry_attemps):
            print(f"Call attempt: {retry_counter}")
            try:
                response = get_model_response(
                    question,
                    provider,
                    model,
                    eval_mode,
                    rest_api_timeout,
                    self._api_client,
                )
                break
            except Exception:
                if retry_counter == retry_attemps - 1:
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

    def _calculate_scores(self, answer, response):
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

    def _get_evaluation_score(self, qna_pool_df):
        """Get response evaluation score."""
        print("Getting evaluation scores...")
        qna_pool_df[
            [
                "cos_score",
                "euc_score",
                "len_score",
                "rougeL_precision",
                "rougeL_recall",
                "rougeL_f1",
            ]
        ] = qna_pool_df.progress_apply(
            lambda row: self._calculate_scores(row.answer, row.response),
            axis=1,
            result_type="expand",
        )
        return qna_pool_df

    def _get_response_with_score(self):
        """Get responses with scores."""
        result_dfs = []
        for provider_model_id in self._args.eval_provider_model_id:
            for eval_mode in self._args.eval_modes:
                print(f"Model evaluation for {provider_model_id}; Mode: {eval_mode}")
                qna_pool_df = self._get_inscope_qna(provider_model_id)
                qna_pool_df = self._get_model_response(
                    qna_pool_df, provider_model_id, eval_mode
                )
                qna_pool_df = self._get_evaluation_score(qna_pool_df)
                qna_pool_df["eval_mode"] = eval_mode
                qna_pool_df["provider_model_id"] = provider_model_id
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
                "doc_page",
                "doc_title",
                "consistency_cutoff",
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

    def _plot_score(self, results_df, score_name):
        """Plot score."""
        _, ax = plt.subplots(figsize=(14, 8))
        ax.set_xlabel(SCORE_DESCRIPTION[score_name])
        ax.set_xlim(0, 1)

        ax.axvline(x=0.25, linewidth=2, color="red")
        ax.axvline(x=0.5, linewidth=2, color="orange")
        ax.axvline(x=0.75, linewidth=2, color="green")

        ax.axvspan(0, 0.25, facecolor="gainsboro")
        ax.axvspan(0.25, 0.5, facecolor="mistyrose")
        ax.axvspan(0.5, 0.75, facecolor="lightyellow")
        ax.axvspan(0.75, 1.0, facecolor="lightgreen")

        ax.grid(True)

        # labels=self._args.eval_provider_model_id
        labels = results_df.columns
        bplot = ax.boxplot(
            results_df,
            patch_artist=True,
            sym=".",
            widths=0.5,
            # tick_labels=labels,
            labels=labels,
            vert=False,
        )
        colors = list(BASE_COLORS.keys())[: len(labels)]
        for patch, color in zip(bplot["boxes"], colors):
            patch.set_facecolor(color)

        plt.yticks(rotation=45)
        plt.savefig(f"{self._result_dir}/model_evaluation_result-{score_name}.png")

    def evaluate_models(self):
        """Evaluate models against groundtruth answer."""
        print("Running model evaluation using groundtruth...")
        result_df = self._get_response_with_score()
        result_df = self._condense_eval_df(result_df)
        result_df.to_csv(f"{self._result_dir}/model_evaluation_result.csv")

        result_df = result_df.groupby(level="query_id").max()

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

            self._plot_score(temp_result_df, score_type)
            summary_score[score_type] = temp_result_df.describe().T.to_dict()

        summary_result = {
            "timestamp": str(datetime.now(UTC)),
            "score": summary_score,
        }
        # Save model evaluation summary report
        with open(f"{self._result_dir}/model_evaluation_summary.json", "w") as f:
            json.dump(summary_result, f)
