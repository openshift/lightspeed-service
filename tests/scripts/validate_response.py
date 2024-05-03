"""Response validation using pre-defined question/answer pair."""

import argparse
import json
import os
import sys
from collections import defaultdict

import requests
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pandas import DataFrame
from scipy.spatial.distance import cosine, euclidean

from tests.e2e.cluster_utils import (
    create_user,
    get_user_token,
    grant_sa_user_access,
)
from tests.e2e.constants import LLM_REST_API_TIMEOUT
from tests.e2e.helper_utils import get_http_client


# TODO: OLS-491 Generate QnA for each model/scenario for evaluation
def _args_parser(args):
    """Arguments parser."""
    parser = argparse.ArgumentParser(description="Response validation module.")
    parser.add_argument(
        "-s",
        "--scenario",
        choices=["with_rag", "without_rag"],
        default="with_rag",
        type=str,
        help="Scenario for which responses will be evaluated.",
    )
    parser.add_argument(
        "-p",
        "--provider",
        default="openai",
        help="Provider name, currently used only to form output file name.",
    )
    parser.add_argument(
        "-m",
        "--model",
        choices=["gpt", "granite"],
        default="gpt",
        type=lambda v: "gpt" if "gpt" in v else "granite",
        help="Model for which responses will be evaluated.",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.7,
        help="Threshold value to be used for similarity score.",
    )
    parser.add_argument(
        "-q",
        "--query_ids",
        nargs="+",
        default=None,
        help="Ids of questions to be validated. Check json file for valid ids.",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        default="tests/test_results",
        help="Result destination.",
    )
    return parser.parse_args(args)


class ResponseValidation:
    """Validate LLM response."""

    def __init__(self):
        """Initialize."""
        self._embedding_model = HuggingFaceEmbedding(
            "sentence-transformers/all-mpnet-base-v2"
        )

    def get_similarity_score(self, response, answer):
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

    def get_response_quality(self, args, qa_pairs, api_client):
        """Get response quality."""
        result_dict = defaultdict(list)

        query_ids = args.query_ids
        if not query_ids:
            query_ids = qa_pairs.keys()

        for query_id in query_ids:
            question = qa_pairs[query_id]["question"]
            answer = qa_pairs[query_id]["answer"]

            response = api_client.post(
                "/v1/query",
                json={"query": question},
                timeout=LLM_REST_API_TIMEOUT,
            )
            if response.status_code != requests.codes.ok:
                raise Exception(response)

            response = response.json()["response"]

            print(f"Calculating score for query: {question}")
            score = self.get_similarity_score(response, answer)

            if score > args.threshold:
                print(f"Response is not as expected for question: {question}")

            result_dict["question"].append(question)
            result_dict["answer"].append(answer)
            result_dict["llm_response"].append(response)
            result_dict["recent_score"].append(score)

        result_df = DataFrame.from_dict(result_dict)
        result_df["threshold"] = args.threshold
        return result_df


def main():
    """Validate LLM response."""
    args = _args_parser(sys.argv[1:])
    print(f"Arguments passed: {args}")

    with open("tests/test_data/question_answer_pair.json") as qna_f:
        qa_pairs = json.load(qna_f)
        qa_pairs = qa_pairs.get(args.model, {}).get(args.scenario, [])

    ols_url = os.getenv("OLS_URL", "http://localhost:8080")
    token = None
    if "localhost" not in ols_url:
        create_user("test-user")
        token = get_user_token("test-user")
        grant_sa_user_access("test-user", "ols-user")
    api_client = get_http_client(ols_url, token)

    result_df = ResponseValidation().get_response_quality(args, qa_pairs, api_client)

    if len(result_df) > 0:
        result_dir = args.out_dir
        os.makedirs(result_dir, exist_ok=True)
        result_file = (
            f"{result_dir}/question_answer_result_"
            f"{args.provider}_{args.model}_{args.scenario}.csv"
        )
        result_df.to_csv(result_file, index=False)
        print(f"Result is saved to {result_file}")

        if result_df.recent_score.max() > args.threshold:
            raise Exception(
                "Response is not matching for question(s):\n"
                f"Please check result in {result_file}."
            )
    else:
        print("No result. Nothing to process.")


if __name__ == "__main__":
    main()
