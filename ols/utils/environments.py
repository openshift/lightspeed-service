"""Environment variables handling."""

import os
import tempfile


def configure_gradio_ui_envs() -> None:
    """Configure GradioUI framework environment variables."""
    # disable Gradio analytics, which calls home to https://api.gradio.app
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "false"

    # Setup config directory for Matplotlib. It will be used to store info
    # about fonts (usually one JSON file) and it really is just temporary
    # storage that can be deleted at any time and recreated later.
    # Fixes: https://issues.redhat.com/browse/OLS-301
    tempdir = os.path.join(tempfile.gettempdir(), "matplotlib")
    os.environ["MPLCONFIGDIR"] = tempdir
