"""Environment variables handling."""

import os
import tempfile
from pathlib import Path

import ols.app.models.config as config_model
from ols.constants import EMBEDDINGS_MODEL_BYOK_SUBDIR


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


def _resolve_embeddings_path(path: Path) -> Path:
    """Resolve embeddings path to model subdirectory when needed."""
    if (path / "config.json").is_file():
        return path
    candidate = path / EMBEDDINGS_MODEL_BYOK_SUBDIR
    if (candidate / "config.json").is_file():
        return candidate
    return path


def configure_hugging_face_envs(ols_config: config_model.OLSConfig) -> None:
    """Configure HuggingFace library environment variables."""
    if (
        ols_config
        and hasattr(ols_config, "reference_content")
        and hasattr(ols_config.reference_content, "embeddings_model_path")
        and ols_config.reference_content.embeddings_model_path
    ):
        resolved = _resolve_embeddings_path(
            Path(ols_config.reference_content.embeddings_model_path)
        )
        os.environ["TRANSFORMERS_CACHE"] = str(resolved)
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
