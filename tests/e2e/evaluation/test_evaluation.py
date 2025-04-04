"""Model response and model evaluation tests."""

from argparse import Namespace

import os
import pytest

from scripts.evaluation.response_evaluation import ResponseEvaluation


def test_model_response(request) -> None:
    """Evaluate model response."""
    args = Namespace(**vars(request.config.option))
    if not args.eval_provider_model_id:
        args.eval_provider_model_id = []
        providers = os.getenv("PROVIDER")
        models = os.getenv("MODEL")
        for i in range(len(providers)):
            args.eval_provider_model_id.append(f"{providers[i]}+{models[i]}")
    args.eval_type = "consistency"

    val_success_flag = ResponseEvaluation(args, pytest.client).validate_response()
    # If flag is False, then response(s) is not consistent,
    # And score is more than cut-off score.
    # Please check eval_result/response_evaluation_* csv file in artifact folder or
    # Check the log to find out exact file path.
    assert val_success_flag


def test_model_evaluation(request) -> None:
    """Evaluate model."""
    # TODO: Use this to assert.
    ResponseEvaluation(request.config.option, pytest.client).evaluate_models()
