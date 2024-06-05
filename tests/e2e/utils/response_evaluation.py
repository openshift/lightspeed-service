"""Response evaluation using pre-defined question/answer pair."""

import json
import os
from collections import defaultdict

import requests
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pandas import DataFrame
from scipy.spatial.distance import cosine, euclidean

from tests.e2e.utils.constants import EVAL_THRESHOLD, LLM_REST_API_TIMEOUT


# TODO: OLS-491 Generate QnA for each model/scenario for evaluation
class ResponseEvaluation:
    """Evaluate LLM response."""

    def __init__(self, eval_args, api_client):
        """Initialize."""
        print(f"Response evaluation arguments: {eval_args}")
        self._args = eval_args
        self._args.eval_model = "gpt" if "gpt" in self._args.eval_model else "granite"
        self._args.eval_threshold = self._args.eval_threshold or EVAL_THRESHOLD

        self._api_client = api_client

        self._embedding_model = HuggingFaceEmbedding(
            "sentence-transformers/all-mpnet-base-v2"
        )

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

    def _get_evaluation_score(self, qa_pairs):
        """Get response evaluation score."""
        result_dict = defaultdict(list)

        query_ids = self._args.eval_query_ids
        if not query_ids:
            query_ids = qa_pairs.keys()

        for query_id in query_ids:
            question = qa_pairs[query_id]["question"]
            answer = qa_pairs[query_id]["answer"]

            response = self._api_client.post(
                "/v1/query",
                json={"query": question},
                timeout=LLM_REST_API_TIMEOUT,
            )
            if response.status_code != requests.codes.ok:
                raise Exception(response)

            response = response.json()["response"].strip()

            print(f"Calculating score for query: {question}")
            score = self._similarity_score(response, answer)

            if score > self._args.eval_threshold:
                print(
                    f"Response is not as expected for question: {question}\n"
                    f"Score: {score} is above cut-off value: {self._args.eval_threshold}"
                )

            result_dict["question"].append(question)
            result_dict["answer"].append(answer)
            result_dict["llm_response"].append(response)
            result_dict["recent_score"].append(score)

        result_df = DataFrame.from_dict(result_dict)
        result_df["threshold"] = self._args.eval_threshold
        return result_df

    def validate_response(self):
        """Validate LLM response."""
        with open("tests/test_data/question_answer_pair.json") as qna_f:
            qa_pairs = json.load(qna_f)
            qa_pairs = qa_pairs.get(self._args.eval_model, {}).get(
                self._args.eval_scenario, []
            )

        result_df = self._get_evaluation_score(qa_pairs)

        if len(result_df) > 0:
            result_dir = self._args.eval_out_dir
            os.makedirs(result_dir, exist_ok=True)
            result_file = (
                f"{result_dir}/response_evaluation_result-"
                f"{self._args.eval_provider}-{self._args.eval_model}-"
                f"{self._args.eval_scenario}.csv"
            )
            result_df.to_csv(result_file, index=False)
            print(f"Result is saved to {result_file}")

            if result_df.recent_score.max() > self._args.eval_threshold:
                # If one of the score is more than threshold,
                # then return False (Failed validation scenario)
                print(
                    "Response is not matching for question(s):\n"
                    f"Please check result in {result_file}."
                )
                return False
        else:
            print("No result. Nothing to process.")

        return True
