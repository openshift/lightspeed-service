"""Driver for evaluation."""

import argparse
import sys

from httpx import Client

from scripts.evaluation.response_evaluation import ResponseEvaluation


def _args_parser(args):
    """Arguments parser."""
    parser = argparse.ArgumentParser(description="Response validation module.")
    parser.add_argument(
        "--eval_provider_model_id",
        nargs="+",
        default=["watsonx+ibm/granite-4-h-small"],
        type=str,
        help="Identifier for Provider/Model to be used for model eval.",
    )
    parser.add_argument(
        "--judge_provider",
        default="ollama",
        type=str,
        help="Provider name for judge model; required for LLM based evaluation",
    )
    parser.add_argument(
        "--judge_model",
        default="llama3.1:latest",
        type=str,
        help="Judge model; required for LLM based evaluation",
    )
    parser.add_argument(
        "--eval_out_dir",
        default=None,
        type=str,
        help="Result destination.",
    )
    parser.add_argument(
        "--eval_query_ids",
        nargs="+",
        default=None,
        help="Ids of questions to be validated. Check json file for valid ids.",
    )
    parser.add_argument(
        "--eval_scenario",
        choices=["with_rag", "without_rag"],
        default="with_rag",
        type=str,
        help="Scenario for which responses will be evaluated.",
    )
    parser.add_argument(
        "--use_default_json_data",
        default=True,
        type=bool,
        help="When True, default data will be used.",
    )
    parser.add_argument(
        "--qna_pool_file",
        default=None,
        type=str,
        help="Additional file having QnA pool in parquet format.",
    )
    parser.add_argument(
        "--eval_type",
        choices=["consistency", "model", "all"],
        default="model",
        type=str,
        help="Evaluation type.",
    )
    parser.add_argument(
        "--eval_metrics",
        nargs="+",
        default=["cos_score"],
        help="Evaluation score/metric.",
    )
    parser.add_argument(
        "--eval_modes",
        nargs="+",
        default=["ols"],
        help="Evaluation modes ex: with just prompt/rag etc.",
    )
    parser.add_argument(
        "--eval_api_url",
        default="http://localhost:8080",
        type=str,
        help="API URL",
    )
    parser.add_argument(
        "--eval_api_token_file",
        default="ols_api_key.txt",
        type=str,
        help="Path to text file with API token (applicable when deployed on cluster)",
    )
    return parser.parse_args(args)


def main():
    """Evaluate response."""
    args = _args_parser(sys.argv[1:])

    client = Client(base_url=args.eval_api_url, verify=False)  # noqa: S501

    if "localhost" not in args.eval_api_url:
        with open(args.eval_api_token_file, mode="r", encoding="utf-8") as t_f:
            token = t_f.read().rstrip()
        client.headers.update({"Authorization": f"Bearer {token}"})

    resp_eval = ResponseEvaluation(args, client)

    match args.eval_type:
        case "consistency":
            resp_eval.validate_response()
        case "model":
            resp_eval.evaluate_models()
        case _:
            resp_eval.validate_response()
            resp_eval.evaluate_models()


if __name__ == "__main__":
    main()
