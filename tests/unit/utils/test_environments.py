"""Unit tests for functions defined in environments.py."""

import os
from unittest.mock import patch

from ols.utils.environments import configure_gradio_ui_envs


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
