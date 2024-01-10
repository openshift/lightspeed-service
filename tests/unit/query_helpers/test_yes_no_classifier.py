"""Unit tests for YesNoClassifier class."""

import pytest

import ols.src.query_helpers.yes_no_classifier
from ols.src.query_helpers.yes_no_classifier import YesNoClassifier
from tests.mock_classes.llm_chain import mock_llm_chain
from tests.mock_classes.llm_loader import mock_llm_loader


@pytest.fixture
def yes_no_classifier():
    """Fixture containing constructed and initialized YesNoClassifier."""
    return YesNoClassifier()


def test_bad_value_response(yes_no_classifier, monkeypatch):
    """Test how YesNoClassifier handles improper responses."""
    # response that isn't 1, 0, or 9 should generate a ValueError
    ml = mock_llm_chain({"text": "default"})

    monkeypatch.setattr(ols.src.query_helpers.yes_no_classifier, "LLMChain", ml)
    monkeypatch.setattr(
        ols.src.query_helpers.yes_no_classifier, "LLMLoader", mock_llm_loader()
    )

    with pytest.raises(ValueError):
        yes_no_classifier.classify(conversation="1234", statement="The sky is blue.")


def test_good_value_response(yes_no_classifier, monkeypatch):
    """Test how YesNoClassifier handles proper responses."""
    # response that is 1, 0, or 9 should return the value
    for x in ["0", "1", "9"]:
        ml = mock_llm_chain({"text": x})

        monkeypatch.setattr(ols.src.query_helpers.yes_no_classifier, "LLMChain", ml)
        monkeypatch.setattr(
            ols.src.query_helpers.yes_no_classifier, "LLMLoader", mock_llm_loader()
        )

        assert yes_no_classifier.classify(
            conversation="1234", statement="The sky is blue."
        ) == int(x)
