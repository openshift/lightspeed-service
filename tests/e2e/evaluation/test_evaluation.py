"""Model response and model evaluation tests."""

# we add new attributes into pytest instance, which is not recognized
# properly by linters
# pyright: reportAttributeAccessIssue=false

import os
from argparse import Namespace

import pytest

from scripts.evaluation.response_evaluation import ResponseEvaluation


def test_model_response(request) -> None:
    """Evaluate model response."""
    args = Namespace(**vars(request.config.option))
    args.eval_provider_model_id = []
    providers = os.getenv("PROVIDER", "").split()
    models = os.getenv("MODEL", "").split()
    for i, provider in enumerate(providers):
        args.eval_provider_model_id.append(f"{provider}+{models[i]}")
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
