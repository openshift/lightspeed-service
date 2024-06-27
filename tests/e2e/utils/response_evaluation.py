"""Response evaluation using pre-defined question/answer pair."""

import json
import os
from collections import defaultdict
from datetime import UTC, datetime

import requests
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pandas import DataFrame, read_csv
from scipy.spatial.distance import cosine, euclidean

from tests.e2e.utils.constants import EVAL_THRESHOLD, LLM_REST_API_TIMEOUT


# TODO: OLS-712 Enrichment of Q+A pairs to contain questions with attachments
class ResponseEvaluation:
    """Evaluate LLM response."""

    def __init__(self, eval_args, api_client):
        """Initialize."""
        print(f"Response evaluation arguments: {eval_args}")
        self._args = eval_args
        self._api_client = api_client

        self._embedding_model = HuggingFaceEmbedding(
            "sentence-transformers/all-mpnet-base-v2"
        )

        with open("tests/test_data/question_answer_pair.json") as qna_f:
            self._qa_pairs = json.load(qna_f)["evaluation"]

    def _get_api_response(self, question, provider, model):
        """Get api response for a question/query."""
        response = self._api_client.post(
            "/v1/query",
            json={
                "query": question,
                "provider": provider,
                "model": model,
            },
            timeout=LLM_REST_API_TIMEOUT,
        )
        if response.status_code != requests.codes.ok:
            raise Exception(response)

        print(f"API request is successful for {provider}+{model}; Query: {question}")
        return response.json()["response"].strip()

    def _get_recent_response(self, recent_resp_df, question, provider, model):
        """Get llm response from the stored data, if available."""
        if recent_resp_df is not None:
            try:
                return recent_resp_df[recent_resp_df.question == question][
                    "llm_response"
                ].iloc[0]
            except IndexError:
                print(
                    "Recent response for query is not found in the file. "
                    "Separate api call is required to get response."
                )
        # Recent response is not found, call api to get response
        return self._get_api_response(question, provider, model)

    def _similarity_score(self, response, answer):
        """Calculate similarity score between two strings."""
        res_vec = self._embedding_model.get_text_embedding(response)
        ans_vec = self._embedding_model.get_text_embedding(answer)

        # Distance score
        cos_score = cosine(res_vec, ans_vec)
        euc_score = euclidean(res_vec, ans_vec)

        # Naive length consideration with reduced weightage.
        len_res, len_ans = len(response), len(answer)
        len_score = (abs(len_res - len_ans) / (len_res + len_ans)) * 0.1

        score = len_score + (cos_score + euc_score) / 2
        # TODO: OLS-409 Use non-contextual score to evaluate response

        print(
            f"cos_score: {cos_score}, "
            f"euc_score: {euc_score}, "
            f"len_score: {len_score}\n"
            f"final_score: {score}"
        )
        return score

    def _get_evaluation_score(self, answer_id):
        """Get response evaluation score."""
        result_dict = defaultdict(list)

        query_ids = self._args.eval_query_ids
        if not query_ids:
            query_ids = self._qa_pairs.keys()

        for query_id in query_ids:
            answer_data = self._qa_pairs[query_id]["answer"][answer_id]
            if not answer_data.get("in_use", True):
                continue

            question = self._qa_pairs[query_id]["question"]
            answers = answer_data["text"]
            eval_threshold = answer_data.get("cutoff_score", EVAL_THRESHOLD)

            response = self._get_api_response(
                question, self._args.eval_provider, self._args.eval_model
            )

            print(f"Calculating score for query: {question}")
            for answer in answers:
                score = self._similarity_score(response, answer)

                if score > eval_threshold:
                    print(
                        f"Response is not as expected for question: {question}\n"
                        f"Score: {score} is above cut-off value: {eval_threshold}"
                    )

                result_dict["eval_id"].append(query_id)
                result_dict["question"].append(question)
                result_dict["answer"].append(answer)
                result_dict["llm_response"].append(response)
                result_dict["consistency_score"].append(score)
                result_dict["cutoff_score"].append(eval_threshold)

        return DataFrame.from_dict(result_dict)

    def validate_response(self):
        """Validate LLM response."""
        answer_id = (
            f"{self._args.eval_provider}+"
            f"{self._args.eval_model}+"
            f"{self._args.eval_scenario}"
        )
        result_df = self._get_evaluation_score(answer_id)

        if len(result_df) > 0:
            result_df["answer_eval_fail_flag"] = (
                result_df.consistency_score > result_df.cutoff_score
            )
            # If none of the answer for any question has score below threshold,
            # then mark as evaluation failure.
            result_df["question_eval_fail_flag"] = result_df.groupby(
                "eval_id"
            ).answer_eval_fail_flag.transform("min")

            result_dir = self._args.eval_out_dir
            os.makedirs(result_dir, exist_ok=True)
            result_file = (
                f"{result_dir}/response_evaluation_result-"
                f"{answer_id.replace('/', '-')}.csv"
            )
            result_df.to_csv(result_file, index=False)
            print(f"Result is saved to {result_file}")

            if result_df.question_eval_fail_flag.max() == 1:
                # If evaluation has failed for any question,
                # then return False (Failed validation scenario)
                print(
                    "Response is not matching for question(s):\n"
                    f"Please check result in {result_file}."
                )
                return False
        else:
            print("No result. Nothing to process.")

        return True

    def evaluate_models(self):
        """Evaluate models against groundtruth answer."""
        print("Running model evaluation using groundtruth...")
        inscope_models = [
            ("bam", "ibm/granite-13b-chat-v2"),
            ("watsonx", "ibm/granite-13b-chat-v2"),
            ("openai", "gpt-3.5-turbo"),
            ("azure_openai", "gpt-3.5-turbo"),
        ]
        answer_id = "ground_truth+with_rag"

        result_df = DataFrame(columns=["eval_id", "question", answer_id])
        for provider_model in inscope_models:
            result_dict = self.evaluate_model(provider_model, answer_id)

            model_result_df = DataFrame.from_dict(result_dict)
            result_df = result_df.merge(
                model_result_df, on=["eval_id", "question", answer_id], how="outer"
            )

        result_df.to_csv(
            f"{self._args.eval_out_dir}/model_evaluation_result.csv", index=False
        )

        average_scores = {}
        for p_m in inscope_models:
            provider_model = "+".join(p_m)
            average_scores[provider_model] = result_df[f"{provider_model}_score"].mean()

        summary_dict = {
            "timestamp": str(datetime.now(UTC)),
            "eval_set": result_df.eval_id.unique().tolist(),
            "avg_similarity_score": average_scores,
        }

        with open(f"{self._args.eval_out_dir}/model_evaluation_summary.json", "w") as f:
            json.dump(summary_dict, f)

    def evaluate_model(self, provider_and_model, answer_id):
        """Evaluate selected provider + model using groundtruth."""
        provider_model = "+".join(provider_and_model)
        provider, model = provider_and_model
        result_dict = defaultdict(list)
        print(f"Model evaluation for {provider_model}")
        try:
            recent_resp_df = read_csv(
                f"{self._args.eval_out_dir}/response_evaluation_result-"
                f"{provider_model.replace('/', '-')}+with_rag.csv"
            )
        except FileNotFoundError:
            print(
                "File with recent model response not found. "
                "Separate api calls are required to get the response."
            )
            recent_resp_df = None

        for query_id in self._qa_pairs.keys():
            question = self._qa_pairs[query_id]["question"]
            answer_data = self._qa_pairs[query_id]["answer"][answer_id]
            answer = answer_data["text"][0]

            response = self._get_recent_response(
                recent_resp_df, question, provider, model
            )
            score = self._similarity_score(response, answer)

            result_dict["eval_id"].append(query_id)
            result_dict["question"].append(question)
            result_dict[answer_id].append(answer)
            result_dict[f"{provider_model}_response"].append(response)
            result_dict[f"{provider_model}_score"].append(score)

        return result_dict
