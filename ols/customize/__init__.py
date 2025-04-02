"""Contains customization packages for individual projects (for prompts/keywords)."""

import importlib
import os

project = os.getenv("PROJECT", "ols")
keywords = importlib.import_module(f"ols.customize.{project}.keywords")
