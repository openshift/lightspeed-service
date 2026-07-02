"""Unit tests for functions defined in environments.py."""

import os
from unittest.mock import patch

from ols.utils.environments import configure_gradio_ui_envs, configure_hugging_face_envs


def test_configure_gradio_ui_envs():
    """Test the function configure_gradio_ui_envs."""
    with patch.dict(os.environ, {"GRADIO_ANALYTICS_ENABLED": "", "MPLCONFIGDIR": ""}):
        # setup before tested function is called
        assert os.environ.get("GRADIO_ANALYTICS_ENABLED", None) == ""
        assert os.environ.get("MPLCONFIGDIR", None) == ""

        # call the tested function
        configure_gradio_ui_envs()

        # expected environment variables
        assert os.environ.get("GRADIO_ANALYTICS_ENABLED", None) == "false"
        assert os.environ.get("MPLCONFIGDIR", None) != ""


def test_configure_hugging_face_envs_sets_offline():
    """Verify configure_hugging_face_envs sets TRANSFORMERS_OFFLINE=1."""
    with patch.dict(os.environ, {"TRANSFORMERS_OFFLINE": ""}):
        assert os.environ.get("TRANSFORMERS_OFFLINE", None) == ""

        configure_hugging_face_envs()

        assert os.environ.get("TRANSFORMERS_OFFLINE", None) == "1"
