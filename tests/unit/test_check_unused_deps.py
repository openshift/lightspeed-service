"""Regression tests for the unused-dependency audit exclusion patterns.

Deptry 0.24 prepends ``^`` to the combined exclusion regex, so ``(^|/)``
cannot match past a parent-directory prefix. The exclusion patterns in
``pyproject.toml`` must keep imports inside nested ``venv`` directories
and nested ``setup.py`` files from masking an otherwise unused dependency.
"""

import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def deptry_section():
    """Extract the [tool.deptry] config verbatim from the real pyproject."""
    text = (ROOT / "pyproject.toml").read_text()
    start = text.index("[tool.deptry]\n")
    end = text.index("\n[tool.", start + 1)
    return text[start:end]


class CheckUnusedDepsExclusionTests(unittest.TestCase):
    """Nested environment paths must not mask unused dependencies."""

    def make_repo(self, masked=None):
        """Create a repository whose only 'requests' import is in a masked path."""
        repo = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (repo / "pyproject.toml").write_text(
            '[project]\nname = "audit-probe"\nversion = "0.0.0"\n'
            'dependencies = ["requests"]\n\n' + deptry_section() + "\n"
        )
        (repo / "ols").mkdir()
        (repo / "ols/tool.py").write_text("# production code without imports\n")
        if masked:
            path = repo / masked
            path.parent.mkdir(parents=True)
            path.write_text("import requests\n")
        return repo

    def run_deptry(self, repo):
        """Run the pinned deptry command exactly as the Makefile does."""
        return subprocess.run(
            [  # noqa: S607 -- resolved from PATH like the Makefile
                "uvx",
                "deptry==0.24.0",
                ".",
            ],
            cwd=repo,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_import_in_nested_venv_does_not_mask_unused_dep(self):
        """Report DEP002 when the only import lives in a nested venv directory."""
        repo = self.make_repo("nested/venv/probe.py")
        result = self.run_deptry(repo)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("DEP002", result.stdout + result.stderr)

    def test_import_in_nested_setup_py_does_not_mask_unused_dep(self):
        """Report DEP002 when the only import lives in a nested setup.py."""
        repo = self.make_repo("nested/setup.py")
        result = self.run_deptry(repo)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("DEP002", result.stdout + result.stderr)

    def test_stdlib_backport_dependency_does_not_report_dep005(self):
        """Keep the audit DEP002-only; DEP005 stays disabled."""
        repo = self.make_repo()
        (repo / "pyproject.toml").write_text(
            (repo / "pyproject.toml")
            .read_text()
            .replace('dependencies = ["requests"]', 'dependencies = ["typing"]')
        )
        (repo / "ols/tool.py").write_text("import typing\n")
        result = self.run_deptry(repo)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_used_dependency_still_passes(self):
        """Keep the audit green when the dependency is genuinely used."""
        repo = self.make_repo()
        (repo / "ols/tool.py").write_text("import requests\n")
        result = self.run_deptry(repo)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
