"""Unit tests for functions defined in environments.py."""

import os
from unittest.mock import patch

from ols.app.models.config import OLSConfig, ReferenceContent
from ols.utils.environments import configure_gradio_ui_envs, configure_hugging_face_envs


@patch.dict(os.environ, {"GRADIO_ANALYTICS_ENABLED": "", "MPLCONFIGDIR": ""})
def test_configure_gradio_ui_envs():
    """Test the function configure_gradio_ui_envs."""
    # setup before tested function is called
    assert os.environ.get("GRADIO_ANALYTICS_ENABLED", None) == ""
    assert os.environ.get("MPLCONFIGDIR", None) == ""

    # call the tested function
    configure_gradio_ui_envs()

    # expected environment variables
    assert os.environ.get("GRADIO_ANALYTICS_ENABLED", None) == "false"
    assert os.environ.get("MPLCONFIGDIR", None) != ""


@patch.dict(os.environ, {"TRANSFORMERS_CACHE": "", "TRANSFORMERS_OFFLINE": ""})
def test_configure_hugging_face_env_no_reference_content_set():
    """Test the function configure_hugging_face_envs."""
    # setup before tested function is called
    assert os.environ.get("TRANSFORMERS_CACHE", None) == ""
    assert os.environ.get("TRANSFORMERS_OFFLINE", None) == ""

    ols_config = OLSConfig()
    ols_config.reference_content = None

    # call the tested function
    configure_hugging_face_envs(ols_config)

    # expected environment variables
    assert os.environ.get("TRANSFORMERS_CACHE", None) == ""
    assert os.environ.get("TRANSFORMERS_OFFLINE", None) == ""


@patch.dict(os.environ, {"TRANSFORMERS_CACHE": "", "TRANSFORMERS_OFFLINE": ""})
def test_configure_hugging_face_env_reference_content_set():
    """Test the function configure_hugging_face_envs."""
    # setup before tested function is called
    assert os.environ.get("TRANSFORMERS_CACHE", None) == ""
    assert os.environ.get("TRANSFORMERS_OFFLINE", None) == ""

    ols_config = OLSConfig()
    ols_config.reference_content = ReferenceContent()
    ols_config.reference_content.embeddings_model_path = "foo"

    # call the tested function
    configure_hugging_face_envs(ols_config)

    # expected environment variables
    assert os.environ.get("TRANSFORMERS_CACHE", None) == "foo"
    assert os.environ.get("TRANSFORMERS_OFFLINE", None) == "1"
