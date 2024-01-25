"""Unit tests for YesNoClassifier class."""

from unittest.mock import patch

import pytest

from ols.src.query_helpers.yes_no_classifier import QueryHelper, YesNoClassifier
from tests.mock_classes.llm_chain import mock_llm_chain
from tests.mock_classes.llm_loader import mock_llm_loader


@pytest.fixture
def yes_no_classifier():
    """Fixture containing constructed and initialized YesNoClassifier."""
    return YesNoClassifier()


def test_is_query_helper_subclass():
    """Test that YesNoClassifier is a subclass of QueryHelper."""
    assert issubclass(YesNoClassifier, QueryHelper)


@patch("ols.src.query_helpers.yes_no_classifier.LLMLoader", new=mock_llm_loader(None))
def test_bad_value_response(yes_no_classifier):
    """Test how YesNoClassifier handles improper responses."""
    # response that isn't 1, 0, or 9 should generate a ValueError
    ml = mock_llm_chain({"text": "default"})

    with patch("ols.src.query_helpers.yes_no_classifier.LLMChain", new=ml):
        with pytest.raises(ValueError, match="Returned response not 0, 1, or 9"):
            yes_no_classifier.classify(
                conversation="1234", statement="The sky is blue."
            )


@patch("ols.src.query_helpers.yes_no_classifier.LLMLoader", new=mock_llm_loader(None))
def test_good_value_response(yes_no_classifier):
    """Test how YesNoClassifier handles proper responses."""
    # response that is 1, 0, or 9 should return the value
    for x in ["0", "1", "9"]:
        ml = mock_llm_chain({"text": x})

        with patch("ols.src.query_helpers.yes_no_classifier.LLMChain", new=ml):
            assert yes_no_classifier.classify(
                conversation="1234", statement="The sky is blue."
            ) == int(x)
