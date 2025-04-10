"""Unit tests for plugins."""

import builtins
import pathlib
import shutil
import tempfile

import pytest

from ols.plugins import _import_modules_from_dir


@pytest.fixture
def temp_module_dir():
    """Create a temporary directory with sample modules."""
    temp_dir = tempfile.mkdtemp()
    module_dir = pathlib.Path(temp_dir)

    # create sample modules that set a flag in builtins
    (module_dir / "mod1.py").write_text("import builtins\nbuiltins.mod1_loaded = True")
    (module_dir / "mod2.py").write_text("import builtins\nbuiltins.mod2_loaded = True")

    # to be ignored - starts with __
    (module_dir / "__mod3.py").write_text(
        "import builtins\nbuiltins.mod3_loaded = True"
    )
    # to be ignored - not a .py file
    (module_dir / "mod4.hi").write_text("file to be ignored")

    yield module_dir

    # teardown
    shutil.rmtree(temp_dir)
    for attr in ["mod1_loaded", "mod2_loaded", "mod3_loaded"]:
        if hasattr(builtins, attr):
            delattr(builtins, attr)


def test_import_modules_from_dir(temp_module_dir):
    """Test importing modules from a directory."""
    assert not hasattr(builtins, "mod1_loaded")
    assert not hasattr(builtins, "mod2_loaded")
    assert not hasattr(builtins, "mod3_loaded")

    _import_modules_from_dir(temp_module_dir)

    assert builtins.mod1_loaded is True
    assert builtins.mod2_loaded is True
    assert not hasattr(builtins, "mod3_loaded")
