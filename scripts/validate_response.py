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

from ols.constants import NO_RAG_CONTENT_RESP


# TODO: Add more question/answer pair
def _args_parser(args):
    """Arguments parser."""
    parser = argparse.ArgumentParser(description="Response validation module.")
    parser.add_argument(
        "-s",
        "--scenario",
        choices=["with_rag", "without_rag"],
        default="with_rag",
        help="Scenario for which responses will be evaluated.",
    )
    parser.add_argument(
        "-m",
        "--model",
        choices=["gpt", "granite"],
        default="gpt",
        help="Model for which responses will be evaluated.",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=lambda val: float(val),
        default=0.25,
        help="Threshold value to be used for similarity score.",
    )
    parser.add_argument(
        "-n",
        "--num_questions",
        type=lambda val: int(val),
        default=2,
        help="Number of questions to be validated.",
    )
    return parser.parse_args(args)


class ResponseValidation:
    """Validate LLM response."""

    def __init__(self):
        """Initialize."""
        self._embedding_model = HuggingFaceEmbedding("BAAI/bge-base-en")

    def get_similarity_score(self, response, answer):
        """Calculate similarity score between two strings."""
        response = response.replace(NO_RAG_CONTENT_RESP, "")
        res_vec = self._embedding_model.get_text_embedding(response)
        ans_vec = self._embedding_model.get_text_embedding(answer)

        # Distance score
        cos_score = cosine(res_vec, ans_vec)
        euc_score = euclidean(res_vec, ans_vec)

        # Naive length consideration with reduced weightage.
        len_res, len_ans = len(response), len(answer)
        len_score = (abs(len_res - len_ans) / (len_res + len_ans)) * 0.1

        score = len_score + (cos_score + euc_score) / 2
        # TODO: Consider both contextual/non-contextual embedding.

        print(
            f"cos_score: {cos_score}, "
            f"euc_score: {euc_score}, "
            f"len_score: {len_score}\n"
            f"final_score: {score}"
        )
        return score

    def get_response_quality(self, args, qa_pairs):
        """Get response quality."""
        result_dict = defaultdict(list)

        for idx in range(min(len(qa_pairs), args.num_questions)):
            question = qa_pairs[idx]["question"]
            answer = qa_pairs[idx]["answer"]

            response = requests.post(
                # API question validator can be disabled.
                "http://localhost:8080/v1/query",
                json={"query": question},
                timeout=90,
            ).json()["response"]

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

    parent_dir = os.path.dirname(os.path.dirname(__file__)) + "/tests"

    with open(f"{parent_dir}/test_data/question_answer_pair.json") as qna_f:
        qa_pairs = json.load(qna_f)
        qa_pairs = qa_pairs.get(args.model, {}).get(args.scenario, [])

    result_df = ResponseValidation().get_response_quality(args, qa_pairs)

    if len(result_df) > 0:
        result_dir = f"{parent_dir}/test_results"
        os.makedirs(result_dir, exist_ok=True)
        result_file = (
            f"{result_dir}/question_answer_result_{args.scenario}_{args.model}.csv"
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
