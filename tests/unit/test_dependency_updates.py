"""Exercise dependency input edits and preference-aware runtime resolution."""

import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
import zipfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "konflux_resolve", ROOT / "scripts/konflux_resolve.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Cannot load dependency resolver")
resolver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resolver)


def temp_directory(test: unittest.TestCase) -> Path:
    """Enter a cleanup-managed temporary directory for a test case."""
    # pylint: disable-next=consider-using-with
    return Path(test.enterContext(tempfile.TemporaryDirectory()))


class DependencyInputTests(unittest.TestCase):
    """Keep project constraints intact while raising a package's minimum version."""

    def setUp(self):
        """Create isolated project inputs."""
        self.directory = temp_directory(self)
        self.project = self.directory / "pyproject.toml"
        self.constraints = self.directory / "requirements.constraints.txt"

    def update(self, name="demo", version="2.0"):
        """Apply an update without running any external resolver."""
        resolver.update_dependency(self.project, name, version)

    def test_core_and_earlier_optional_group_are_both_updated(self):
        """Preserve TOML validity when matching arrays occur in either order."""
        self.project.write_text(
            '[project.optional-dependencies]\ncpu = ["Demo>=1"]\n'
            '[project]\nname = "example"\ndependencies = [\n'
            '  "demo>=1,<3", # keep this comment\n]\n',
            encoding="utf-8",
        )
        self.update()
        content = self.project.read_text(encoding="utf-8")
        project = tomllib.loads(content)["project"]
        self.assertEqual(project["dependencies"], ["demo>=2.0,<3"])
        self.assertEqual(project["optional-dependencies"]["cpu"], ["Demo>=2.0"])
        self.assertIn("# keep this comment", content)
        self.assertFalse(self.constraints.exists())

    def test_bare_requirement_preserves_extras_and_marker(self):
        """Keep extras and environment markers on updated dependencies."""
        self.project.write_text(
            "[project]\ndependencies = ['Demo_Name[extra]; python_version < \"3.13\"']\n",
            encoding="utf-8",
        )
        self.update("demo.name")
        dependency = tomllib.loads(self.project.read_text(encoding="utf-8"))["project"][
            "dependencies"
        ][0]
        self.assertIn("Demo_Name[extra]>=2.0", dependency)
        self.assertIn('python_version < "3.13"', dependency)

    def test_transitive_update_does_not_create_persistent_inputs(self):
        """Leave project inputs unchanged for a transitive update."""
        self.project.write_text(
            '[project]\ndependencies = ["other"]\n', encoding="utf-8"
        )
        original = self.project.read_bytes()
        self.update()
        self.assertEqual(self.project.read_bytes(), original)
        self.assertFalse(self.constraints.exists())

    def test_incompatible_bounds_leave_all_inputs_unchanged(self):
        """Reject a requested floor excluded by an explicit compatibility bound."""
        self.project.write_text(
            '[project]\ndependencies = ["demo>=1,<2"]\n', encoding="utf-8"
        )
        self.constraints.write_text("demo>=1\n", encoding="utf-8")
        original = self.project.read_bytes(), self.constraints.read_bytes()
        with self.assertRaisesRegex(ValueError, "compatibility"):
            self.update()
        self.assertEqual(
            original, (self.project.read_bytes(), self.constraints.read_bytes())
        )

    def test_invalid_version_or_url_leaves_inputs_unchanged(self):
        """Reject unsupported updates before writing any input file."""
        self.project.write_text(
            '[project]\ndependencies = ["demo @ https://example.test/demo.whl"]\n',
            encoding="utf-8",
        )
        original = self.project.read_bytes()
        for version in ("not-a-version", "2.0"):
            with self.subTest(version=version), self.assertRaises(ValueError):
                self.update(version=version)
            self.assertEqual(original, self.project.read_bytes())
        self.assertFalse(self.constraints.exists())

    def test_targeted_update_does_not_modify_deliberate_overrides(self):
        """Keep explicit override policy separate from automatically requested minima."""
        self.project.write_text(
            '[project]\ndependencies = ["other"]\n', encoding="utf-8"
        )
        overrides = self.directory / "requirements.overrides.txt"
        overrides.write_text("# deliberate policy\ndemo>=1\n", encoding="utf-8")
        original = overrides.read_bytes()
        self.update()
        self.assertEqual(overrides.read_bytes(), original)
        self.assertFalse(self.constraints.exists())

    def test_requested_minimum_below_compatible_release_keeps_that_range(self):
        """A minimum below an existing compatible range is not a conflict."""
        self.project.write_text(
            '[project]\ndependencies = ["demo~=3.0"]\n', encoding="utf-8"
        )
        self.update(version="2.0")
        self.assertIn("~=3.0", self.project.read_text(encoding="utf-8"))

    def test_wildcard_equality_is_preserved(self):
        """Raising a floor must not open a restricted release series."""
        for minimum in ("0.5", "1.5"):
            with self.subTest(minimum=minimum):
                updated = resolver.bump_requirement("demo==1.*", minimum)
                self.assertIn("==1.*", updated)
                self.assertIn(f">={minimum}", updated)

    def test_incompatible_equality_leaves_project_unchanged(self):
        """Equality bounds are compatibility restrictions, not disposable pins."""
        for bound in ("==1.*", "==1.0", "===1.0"):
            with self.subTest(bound=bound):
                self.project.write_text(
                    f'[project]\ndependencies = ["demo{bound}"]\n', encoding="utf-8"
                )
                original = self.project.read_bytes()
                with self.assertRaisesRegex(ValueError, "compatibility"):
                    self.update(version="2.0")
                self.assertEqual(self.project.read_bytes(), original)

    def test_stronger_existing_floor_is_not_lowered(self):
        """Keep a minimum that is already higher than the requested floor."""
        self.project.write_text(
            '[project]\ndependencies = ["demo>=3"]\n', encoding="utf-8"
        )
        self.update()
        self.assertEqual(
            tomllib.loads(self.project.read_text(encoding="utf-8"))["project"][
                "dependencies"
            ],
            ["demo>=3"],
        )


class BuildDiscoveryTests(unittest.TestCase):
    """Discover only reachable source-build requirements and their dependencies."""

    def test_dependency_annotations_include_multiple_parents(self):
        """Read the dependency graph emitted by the forked compiler."""
        result = resolver._parse_uv_output(
            "helper==1.0\n    # via\n    #   builder\n    #   other[extra]\n"
            "    # from https://pypi.org/simple/\n"
        )
        self.assertEqual(result["helper"]["parents"], ["builder", "other"])

    def test_recursion_follows_build_tools_transitive_source_dependencies(self):
        """Inspect source dependencies of selected build tools, not just top-level sdists."""

        def info(version="1", parents=()):
            return {
                "version": version,
                "index": "https://pypi.org/simple",
                "parents": list(parents),
            }

        combined = {
            "runtime": info(),
            "builder": info(),
            "helper": info(parents=("builder",)),
            "orphan": info(),
        }
        pypi = mock.Mock()
        pypi.get_package_info.return_value = {"1": {"has_sdist": True}}
        requirements = {
            "runtime": ["builder>=1"],
            "builder": [],
            "helper": ["compiler>=2"],
        }
        with mock.patch.object(
            resolver,
            "discover_build_requirements",
            side_effect=lambda name, version: requirements[name],
        ):
            roots = resolver._collect_build_requirements(
                {"runtime": combined["runtime"]},
                combined,
                (),
                "https://rhoai.test",
                set(),
                pypi,
            )
        self.assertEqual(roots, ["builder>=1", "compiler>=2"])
        self.assertNotIn(mock.call("orphan"), pypi.get_package_info.call_args_list)

    def test_minimum_is_forwarded_through_every_resolution_round(self):
        """Apply the temporary minimum to joint solves and runtime projections."""
        selected = {
            "demo": {"version": "2", "index": "https://rhoai.test", "parents": []}
        }
        with mock.patch.object(
            resolver, "uv_resolve", return_value=selected
        ) as compile_:
            resolver.resolve_environment(
                "3.12",
                "https://rhoai.test",
                "",
                (),
                (),
                set(),
                mock.Mock(),
                "demo",
                minimums=("demo>=2",),
            )
        self.assertGreater(len(compile_.call_args_list), 2)
        for call in compile_.call_args_list:
            self.assertEqual(call.kwargs["minimums"], ("demo>=2",))

    def test_discovery_cycle_fails_instead_of_publishing_an_unstable_solution(self):
        """Stop when repeated build requirements force an oscillating selection."""
        calls = 0

        def compile_environment(*_args, **kwargs):
            nonlocal calls
            if kwargs.get("constraints"):
                version = kwargs["constraints"]["demo"]
            else:
                calls += 1
                version = str(calls % 2 + 1)
            return {
                "demo": {
                    "version": version,
                    "index": "https://rhoai.test",
                    "parents": [],
                }
            }

        with mock.patch.object(resolver, "uv_resolve", side_effect=compile_environment):
            with self.assertRaisesRegex(RuntimeError, "did not converge"):
                resolver.resolve_environment(
                    "3.12", "https://rhoai.test", "", (), (), set(), mock.Mock()
                )


class OutputAndCliTests(unittest.TestCase):
    """Publish only compatible selected versions and validate the CLI early."""

    def test_containerfile_installs_prefetched_runtime_pypi_wheels(self):
        """Install the PyPI-wheel lock alongside the RHOAI and source locks."""
        containerfile = (ROOT / "Containerfile").read_text(encoding="utf-8")
        self.assertIn(".konflux/requirements.hashes.wheel.pypi.txt", containerfile)
        self.assertIn("-r requirements.hashes.wheel.pypi.txt", containerfile)

    def test_cli_requires_package_and_version_together(self):
        """Reject incomplete targeted updates before reading project inputs."""
        with tempfile.TemporaryDirectory() as directory:
            for arguments in (("--package", "demo"), ("--version", "2.0")):
                result = (
                    subprocess.run(  # noqa: S603 -- fixed interpreter and arguments
                        [
                            sys.executable,
                            str(ROOT / "scripts/konflux_resolve.py"),
                            "--profile",
                            "cpu",
                            *arguments,
                        ],
                        cwd=directory,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("together", result.stderr)

    def test_build_overlap_is_not_written_at_a_second_version(self):
        """Put runtime artifacts in runtime files and build-only wheels in hermetic output."""
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(resolver, "KONFLUX_DIR", directory):
                packages = {
                    name: {
                        "version": version,
                        "source": "rhoai",
                        "platforms": {"any": ("wheel", "abc")},
                    }
                    for name, version in (("shared", "3.0"), ("builder", "2.0"))
                }
                resolver.publish_environment(
                    {"shared": packages["shared"]},
                    packages,
                    set(),
                    (),
                    "https://rhoai.test/simple/",
                    "",
                )
            self.assertIn(
                "shared==3.0",
                (Path(directory) / "requirements.hashes.wheel.txt").read_text(
                    encoding="utf-8"
                ),
            )
            self.assertIn(
                "builder==2.0",
                (Path(directory) / "requirements.hermetic.txt").read_text(
                    encoding="utf-8"
                ),
            )
            self.assertNotIn(
                "shared==",
                (Path(directory) / "requirements-build.txt").read_text(
                    encoding="utf-8"
                ),
            )

    def test_build_requirements_file_has_no_index_url(self):
        """Keep requirements-build.txt index-free so Hermeto stays PyPI-only."""
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(resolver, "KONFLUX_DIR", directory):
                runtime = {
                    "demo": {
                        "version": "1.0",
                        "source": "rhoai",
                        "platforms": {"any": ("wheel", "abc")},
                    }
                }
                combined = {
                    **runtime,
                    "builder": {
                        "version": "2.0",
                        "source": "pypi",
                        "has_sdist": True,
                        "sdist_hashes": ["deadbeef"],
                        "wheel_hashes": [],
                    },
                }
                resolver.publish_environment(
                    runtime,
                    combined,
                    set(),
                    (),
                    "https://rhoai.test/simple/",
                    "",
                )
            build = (Path(directory) / "requirements-build.txt").read_text(
                encoding="utf-8"
            )
            self.assertIn("builder==2.0", build)
            self.assertNotIn("--index-url", build)
            wheel = (Path(directory) / "requirements.hashes.wheel.txt").read_text(
                encoding="utf-8"
            )
            self.assertIn("--index-url https://rhoai.test/simple/", wheel)

    def test_missing_hash_does_not_publish_partial_outputs(self):
        """Complete generation in scratch space before replacing committed files."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "requirements.hashes.wheel.txt"
            output.write_text("original\n", encoding="utf-8")
            package = {
                "version": "1",
                "source": "rhoai",
                "platforms": {"any": ("wheel", "")},
            }
            with (
                mock.patch.object(resolver, "KONFLUX_DIR", directory),
                self.assertRaisesRegex(RuntimeError, "No hashes"),
            ):
                resolver.publish_environment(
                    {"demo": package},
                    {"demo": package},
                    set(),
                    (),
                    "https://rhoai.test/simple/",
                    "",
                )
            self.assertEqual(output.read_text(encoding="utf-8"), "original\n")

    def test_suffix_cannot_escape_the_output_directory(self):
        """Reject a profile suffix that would redirect generated outputs."""
        with tempfile.TemporaryDirectory() as parent:
            konflux = Path(parent) / "konflux"
            konflux.mkdir()
            package = {
                "version": "1.0",
                "source": "rhoai",
                "platforms": {"any": ("wheel", "abc")},
            }
            with (
                mock.patch.object(resolver, "KONFLUX_DIR", str(konflux)),
                self.assertRaisesRegex(RuntimeError, "path separators"),
            ):
                resolver.publish_environment(
                    {"demo": package},
                    {"demo": package},
                    set(),
                    (),
                    "https://rhoai.test/simple/",
                    "/../../escaped",
                )
            self.assertEqual(list(konflux.iterdir()), [])
            self.assertFalse((Path(parent) / "escaped.txt").exists())

    def test_failed_update_restores_pyproject_toml(self):
        """Restore pyproject.toml when resolution fails after the floor edit."""
        directory = temp_directory(self)
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(directory)
        original = b'[project]\nname = "example"\ndependencies = ["demo>=1"]\n'
        project = Path("pyproject.toml")
        project.write_bytes(original)
        konflux = directory / ".konflux"
        konflux.mkdir()
        (konflux / "profiles.toml").write_text(
            "[common]\n"
            f'python_version = "{sys.version_info.major}.{sys.version_info.minor}"\n'
            'platforms = ["x86_64"]\n'
            "bootstrap_packages = []\n"
            "\n"
            "[profiles.cpu]\n"
            'rhoai_index_url = "https://rhoai.test/simple/"\n',
            encoding="utf-8",
        )
        (konflux / "pypi_wheel_only.txt").write_text("", encoding="utf-8")
        arguments = argparse.Namespace(profile="cpu", package="demo", version="2.0")

        def fail_after_floor_edit(*_args, **_kwargs):
            if "demo>=2" not in project.read_text(encoding="utf-8"):
                self.fail("floor edit missing when resolution started")
            raise RuntimeError("boom")

        with (
            mock.patch.object(resolver, "KONFLUX_DIR", str(konflux)),
            mock.patch.object(
                resolver, "resolve_environment", side_effect=fail_after_floor_edit
            ),
            self.assertRaisesRegex(RuntimeError, "boom"),
        ):
            resolver._run_update(arguments)
        self.assertEqual(project.read_bytes(), original)


class RuntimePreferenceTests(unittest.TestCase):
    """Pass soft preferences to the forked uv without freezing dependencies."""

    def setUp(self):
        """Provide isolated committed pins and a container command boundary."""
        directory = temp_directory(self)
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(directory)
        konflux = Path(".konflux")
        konflux.mkdir()
        (konflux / "requirements.hashes.wheel.txt").write_text(
            "demo==1.0 \\\n    --hash=sha256:abc\nother==3.0\n", encoding="utf-8"
        )
        (konflux / "requirements.hashes.source.txt").write_text(
            "dependency==2.0\n", encoding="utf-8"
        )
        self.runtime = mock.patch.object(
            resolver, "_container_runtime", return_value="podman"
        )
        self.runtime.start()
        self.addCleanup(self.runtime.stop)

    def test_targeted_compile_seeds_existing_versions_as_preferences(self):
        """Use the fork's output reuse and upgrade only the requested root."""
        captured = {}
        Path("requirements.constraints.txt").write_text("demo<2\n", encoding="utf-8")
        Path(".konflux/requirements.overrides.txt").write_text(
            "other==3.0\n", encoding="utf-8"
        )

        def run(command, **_kwargs):
            captured["command"] = command
            volumes = [
                command[i + 1] for i, value in enumerate(command) if value == "--volume"
            ]
            scratch = next(
                volume.split(":")[0] for volume in volumes if ":/resolution:" in volume
            )
            output = Path(scratch) / "resolved.txt"
            captured["seed"] = output.read_text(encoding="utf-8")
            captured["minimums"] = (Path(scratch) / "minimums.txt").read_text(
                encoding="utf-8"
            )
            output.write_text(
                "demo==2.0\n    # from https://rhoai.test/simple/\nother==3.0\n",
                encoding="utf-8",
            )
            captured["scratch"] = scratch
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        with mock.patch.object(resolver.subprocess, "run", side_effect=run):
            result = resolver.uv_resolve(
                "3.12",
                "https://rhoai.test/simple/",
                "",
                upgrade_package="demo",
                minimums=("demo>=2.0",),
            )
        command = captured["command"]
        self.assertIn(resolver.UV_IMAGE, command)
        self.assertEqual(command[command.index("--index-strategy") + 1], "prefer-index")
        self.assertEqual(command[command.index("--upgrade-package") + 1], "demo")
        self.assertEqual(
            command[command.index("--constraints") + 1],
            "/resolution/minimums.txt",
        )
        self.assertEqual(
            command[command.index("--override") + 1],
            ".konflux/requirements.overrides.txt",
        )
        self.assertEqual(captured["minimums"], "demo>=2.0\n")
        self.assertNotIn("./requirements.constraints.txt", command)
        self.assertNotIn("--upgrade", command)
        self.assertIn("other==3.0", captured["seed"])
        self.assertIn("dependency==2.0", captured["seed"])
        self.assertEqual(result["demo"]["version"], "2.0")
        self.assertFalse(Path(captured["scratch"]).exists())

    def test_full_refresh_ignores_persistent_constraints(self):
        """Keep the removed control out of ordinary full refreshes."""
        Path("requirements.constraints.txt").write_text("demo<2\n", encoding="utf-8")

        def run(command, **_kwargs):
            self.assertNotIn("./requirements.constraints.txt", command)
            volumes = [command[i + 1] for i, v in enumerate(command) if v == "--volume"]
            scratch = next(v.split(":")[0] for v in volumes if ":/resolution:" in v)
            (Path(scratch) / "resolved.txt").write_text("demo==5.0\n", encoding="utf-8")

        with mock.patch.object(resolver.subprocess, "run", side_effect=run):
            resolver.uv_resolve("3.12", "https://rhoai.test/simple/", "")

    def test_full_refresh_does_not_reuse_committed_pins(self):
        """Start a full refresh without existing version preferences."""

        def run(command, **_kwargs):
            if "--output-file" in command:
                volumes = [
                    command[i + 1]
                    for i, value in enumerate(command)
                    if value == "--volume"
                ]
                scratch = next(v.split(":")[0] for v in volumes if ":/resolution:" in v)
                output = Path(scratch) / "resolved.txt"
                self.assertFalse(output.exists())
                output.write_text("demo==5.0\n", encoding="utf-8")
            return subprocess.CompletedProcess(
                command, 0, stdout="demo==5.0\n", stderr=""
            )

        with mock.patch.object(resolver.subprocess, "run", side_effect=run):
            result = resolver.uv_resolve("3.12", "https://rhoai.test/simple/", "")
        self.assertEqual(result["demo"]["version"], "5.0")


@unittest.skipUnless(
    os.environ.get("OLS_TEST_RESOLVER_IMAGE") == "1", "requires forked uv image"
)
class ForkedUvIntegrationTests(unittest.TestCase):
    """Resolve deterministic packages using the real prefer-index image."""

    def setUp(self):
        """Build two local indexes and seed the committed runtime pins."""
        directory = temp_directory(self)
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(directory)
        konflux = Path(".konflux")
        konflux.mkdir()
        self.wheel("ols-target", "1.0", ["ols-dependency>=1"])
        self.wheel("ols-target", "2.0", ["ols-dependency>=2"])
        self.wheel("ols-dependency", "1.0")
        self.wheel("ols-dependency", "2.0")
        self.wheel("ols-unrelated", "1.0")
        self.wheel("ols-unrelated", "2.0")
        self.wheel("ols-unrelated", "9.0", index="fallback")
        Path("pyproject.toml").write_text(
            '[project]\nname = "fixture"\nversion = "0"\n'
            'dependencies = ["ols-target>=2", "ols-unrelated"]\n',
            encoding="utf-8",
        )
        (konflux / "requirements.hashes.wheel.txt").write_text(
            "ols-target==1.0\nols-dependency==1.0\nols-unrelated==1.0\n",
            encoding="utf-8",
        )
        # Substitute only the public fallback endpoint; execute the real container command.
        real_run = subprocess.run

        def local_indexes(command, check=False, **kwargs):
            command = list(command)
            command[command.index("--default-index") + 1] = "file:///io/fallback/"
            return real_run(command, check=check, **kwargs)

        patch = mock.patch.object(resolver.subprocess, "run", side_effect=local_indexes)
        patch.start()
        self.addCleanup(patch.stop)

    def wheel(self, name, version, dependencies=(), index="primary"):
        """Write a minimal universal wheel and its simple-index link."""
        normalized = name.replace("-", "_")
        directory = Path(index) / name
        directory.mkdir(parents=True, exist_ok=True)
        filename = f"{normalized}-{version}-py3-none-any.whl"
        metadata = (
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
            "Requires-Python: >=3.12\n"
        )
        metadata += "".join(
            f"Requires-Dist: {dependency}\n" for dependency in dependencies
        )
        with zipfile.ZipFile(directory / filename, "w") as wheel:
            wheel.writestr(f"{normalized}-{version}.dist-info/METADATA", metadata)
            wheel.writestr(
                f"{normalized}-{version}.dist-info/WHEEL",
                "Wheel-Version: 1.0\nTag: py3-none-any\n",
            )
            wheel.writestr(f"{normalized}-{version}.dist-info/RECORD", "")
        with (directory / "index.html").open("a", encoding="utf-8") as stream:
            stream.write(f'<a href="{filename}">{filename}</a>\n')

    def test_transitive_minimum_does_not_override_parent_compatibility(self):
        """Reject a transitive floor when the installed parent's range excludes it."""
        self.wheel("ols-parent", "1.0", ["ols-dependency<2"])
        Path("pyproject.toml").write_text(
            '[project]\nname = "fixture"\nversion = "0"\n'
            'dependencies = ["ols-parent==1.0"]\n',
            encoding="utf-8",
        )
        resolver.update_dependency(
            Path("pyproject.toml"),
            "ols-dependency",
            "2.0",
        )
        with self.assertRaisesRegex(RuntimeError, "No solution") as failure:
            resolver.uv_resolve(
                "3.12",
                "file:///io/primary/",
                "",
                upgrade_package="ols-dependency",
                minimums=("ols-dependency>=2.0",),
            )
        self.assertIn("ols-parent", str(failure.exception))

    def test_transitive_minimum_can_upgrade_the_blocking_parent(self):
        """Permit an available compatible parent upgrade rather than bypassing its range."""
        self.wheel("ols-parent", "1.0", ["ols-dependency<2"])
        self.wheel("ols-parent", "2.0", ["ols-dependency>=2"])
        Path("pyproject.toml").write_text(
            '[project]\nname = "fixture"\nversion = "0"\n'
            'dependencies = ["ols-parent", "ols-unrelated"]\n',
            encoding="utf-8",
        )
        Path(".konflux/requirements.hashes.wheel.txt").write_text(
            "ols-parent==1.0\nols-dependency==1.0\nols-unrelated==1.0\n",
            encoding="utf-8",
        )
        resolver.update_dependency(
            Path("pyproject.toml"),
            "ols-dependency",
            "2.0",
        )
        result = resolver.uv_resolve(
            "3.12",
            "file:///io/primary/",
            "",
            upgrade_package="ols-dependency",
            minimums=("ols-dependency>=2.0",),
        )
        self.assertEqual(result["ols-parent"]["version"], "2.0")
        self.assertEqual(result["ols-dependency"]["version"], "2.0")
        self.assertEqual(result["ols-unrelated"]["version"], "1.0")
        self.assertFalse(Path("requirements.constraints.txt").exists())
        self.assertFalse(Path(".konflux/requirements.overrides.txt").exists())

    def test_requested_minimum_intersects_deliberate_overrides(self):
        """Do not let configured overrides erase an additive security minimum."""
        Path(".konflux/requirements.overrides.txt").write_text(
            "ols-dependency==1.0\n", encoding="utf-8"
        )
        resolver.update_dependency(
            Path("pyproject.toml"),
            "ols-dependency",
            "2.0",
        )
        with self.assertRaisesRegex(RuntimeError, "No solution"):
            resolver.uv_resolve(
                "3.12",
                "file:///io/primary/",
                "",
                upgrade_package="ols-dependency",
                minimums=("ols-dependency>=2.0",),
            )
        self.assertEqual(
            Path(".konflux/requirements.overrides.txt").read_text(encoding="utf-8"),
            "ols-dependency==1.0\n",
        )

    def test_required_dependency_moves_but_unrelated_pin_stays(self):
        """Permit a required upgrade without refreshing an unrelated package."""
        result = resolver.uv_resolve(
            "3.12", "file:///io/primary/", "", upgrade_package="ols-target"
        )
        self.assertEqual(
            {name: info["version"] for name, info in result.items()},
            {"ols-target": "2.0", "ols-dependency": "2.0", "ols-unrelated": "1.0"},
        )

    def test_successive_targets_use_current_pins_without_persistent_constraints(self):
        """Reuse the preceding result without remembering earlier requested minimums."""
        first = resolver.uv_resolve(
            "3.12",
            "file:///io/primary/",
            "",
            upgrade_package="ols-target",
            minimums=("ols-target>=2.0",),
        )
        Path(".konflux/requirements.hashes.wheel.txt").write_text(
            "".join(f"{name}=={info['version']}\n" for name, info in first.items()),
            encoding="utf-8",
        )
        second = resolver.uv_resolve(
            "3.12",
            "file:///io/primary/",
            "",
            upgrade_package="ols-unrelated",
            minimums=("ols-unrelated>=2.0",),
        )
        self.assertEqual(second["ols-target"]["version"], "2.0")
        self.assertEqual(second["ols-unrelated"]["version"], "2.0")
        self.assertFalse(Path("requirements.constraints.txt").exists())

    def test_full_refresh_ignores_old_persistent_constraints(self):
        """Ignore even conflicting entries from the removed control."""
        Path("requirements.constraints.txt").write_text(
            "ols-target<2\n", encoding="utf-8"
        )
        result = resolver.uv_resolve("3.12", "file:///io/primary/", "")
        self.assertEqual(result["ols-target"]["version"], "2.0")

    def test_targeted_runtime_update_is_idempotent(self):
        """Reusing the newly selected pins leaves the package versions unchanged."""
        first = resolver.uv_resolve(
            "3.12", "file:///io/primary/", "", upgrade_package="ols-target"
        )
        pins = {name: info["version"] for name, info in first.items()}
        Path(".konflux/requirements.hashes.wheel.txt").write_text(
            "".join(f"{name}=={version}\n" for name, version in sorted(pins.items())),
            encoding="utf-8",
        )
        second = resolver.uv_resolve(
            "3.12", "file:///io/primary/", "", upgrade_package="ols-target"
        )
        self.assertEqual(pins, {name: info["version"] for name, info in second.items()})

    def test_full_refresh_still_prefers_primary_index(self):
        """Do not replace prefer-index with upstream best-version selection."""
        result = resolver.uv_resolve("3.12", "file:///io/primary/", "")
        self.assertEqual(result["ols-unrelated"]["version"], "2.0")

    def test_build_floor_moves_existing_tool_without_refreshing_other_tools(self):
        """Resolve increased build requirements instead of restoring stale pins."""
        self.wheel("ols-builder", "1.0")
        self.wheel("ols-builder", "2.0")
        self.wheel("ols-build-helper", "1.0")
        self.wheel("ols-build-helper", "2.0")
        self.wheel("ols-new-tool", "1.0")
        Path(".konflux/requirements-build.txt").write_text(
            "ols-builder==1.0\nols-build-helper==1.0\n", encoding="utf-8"
        )
        source = {"ols-target": ["ols-builder>=2", "ols-build-helper", "ols-new-tool"]}
        pypi = mock.Mock()
        pypi.get_package_info.side_effect = lambda name: {
            "2.0": {"has_sdist": name == "ols-target"}
        }
        # Classify the target as a source package, while tools come from the primary index.
        self.source_target()
        with mock.patch.object(
            resolver, "discover_build_requirements", side_effect=lambda n, v: source[n]
        ):
            runtime, combined = resolver.resolve_environment(
                "3.12", "file:///io/primary/", "", (), (), set(), pypi, "ols-target"
            )
        self.assertEqual(combined["ols-builder"]["version"], "2.0")
        self.assertEqual(combined["ols-build-helper"]["version"], "1.0")
        self.assertIn("ols-new-tool", combined)
        self.assertNotIn("ols-builder", runtime)

    def source_target(self):
        """Place the fixed target exclusively on the fallback index."""
        self.wheel("ols-target", "2.0", ["ols-dependency>=2"], index="fallback")
        (Path("primary/ols-target") / "index.html").unlink()
        # Keep the primary package page empty so the fork falls back for this package.
        (Path("primary/ols-target") / "index.html").write_text("", encoding="utf-8")

    def test_build_runtime_overlap_uses_one_compatible_version(self):
        """Revisit runtime selection when a build requirement needs a newer shared package."""
        self.source_target()
        self.wheel("ols-dependency", "3.0")
        Path(".konflux/requirements.hashes.wheel.txt").write_text(
            "ols-target==1.0\nols-dependency==2.0\nols-unrelated==1.0\n",
            encoding="utf-8",
        )
        pypi = mock.Mock()
        pypi.get_package_info.return_value = {"2.0": {"has_sdist": True}}
        with mock.patch.object(
            resolver, "discover_build_requirements", return_value=["ols-dependency>=3"]
        ):
            runtime, combined = resolver.resolve_environment(
                "3.12", "file:///io/primary/", "", (), (), set(), pypi, "ols-target"
            )
        self.assertEqual(runtime["ols-dependency"]["version"], "3.0")
        self.assertEqual(combined["ols-dependency"]["version"], "3.0")
        self.assertEqual(runtime["ols-unrelated"]["version"], "1.0")

    def test_build_target_can_move_its_source_parent(self):
        """Reconsider a source version rather than treating its build bounds as policy."""
        self.wheel("ols-source", "1.0", index="fallback")
        self.wheel("ols-source", "2.0", index="fallback")
        self.wheel("ols-source", "3.0", index="fallback")
        self.wheel("ols-builder", "1.0")
        self.wheel("ols-builder", "2.0")
        Path("pyproject.toml").write_text(
            '[project]\nname="fixture"\nversion="0"\n'
            'dependencies=["ols-source", "ols-unrelated"]\n',
            encoding="utf-8",
        )
        Path(".konflux/requirements.hashes.wheel.txt").write_text(
            "ols-unrelated==1.0\n", encoding="utf-8"
        )
        Path(".konflux/requirements.hashes.source.txt").write_text(
            "ols-source==1.0\n", encoding="utf-8"
        )
        Path(".konflux/requirements-build.txt").write_text(
            "ols-builder==1.0\n", encoding="utf-8"
        )
        pypi = mock.Mock()
        pypi.get_package_info.return_value = {
            version: {"has_sdist": True} for version in ("1.0", "2.0", "3.0")
        }
        with mock.patch.object(
            resolver,
            "discover_build_requirements",
            side_effect=lambda name, version: [
                "ols-builder>=2" if version == "2.0" else "ols-builder<2"
            ],
        ):
            runtime, combined = resolver.resolve_environment(
                "3.12",
                "file:///io/primary/",
                "",
                (),
                (),
                set(),
                pypi,
                "ols-builder",
                minimums=("ols-builder>=2",),
            )
            self.assertEqual(runtime["ols-source"]["version"], "2.0")
            self.assertEqual(combined["ols-builder"]["version"], "2.0")
            self.assertEqual(runtime["ols-unrelated"]["version"], "1.0")
            # A full refresh must also reconsider an incompatible latest source.
            runtime, combined = resolver.resolve_environment(
                "3.12",
                "file:///io/primary/",
                "",
                (),
                (),
                set(),
                pypi,
                minimums=("ols-builder>=2",),
            )
            self.assertEqual(runtime["ols-source"]["version"], "2.0")
            self.assertEqual(runtime["ols-unrelated"]["version"], "2.0")
            # Explicit project bounds must still prevent the update.
            project = Path("pyproject.toml")
            project.write_text(
                project.read_text(encoding="utf-8").replace(
                    '"ols-source"', '"ols-source<2"'
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "No solution"):
                resolver.resolve_environment(
                    "3.12",
                    "file:///io/primary/",
                    "",
                    (),
                    (),
                    set(),
                    pypi,
                    "ols-builder",
                    minimums=("ols-builder>=2",),
                )

    def test_bootstrap_versions_are_reused_when_compatible(self):
        """Do not refresh bootstrap tools just because a newer version exists."""
        self.wheel("ols-bootstrap", "1.0")
        self.wheel("ols-bootstrap", "2.0")
        Path(".konflux/requirements.hermetic.txt").write_text(
            "ols-bootstrap==1.0\n", encoding="utf-8"
        )
        runtime, combined = resolver.resolve_environment(
            "3.12",
            "file:///io/primary/",
            "",
            (),
            ("ols-bootstrap",),
            set(),
            mock.Mock(),
            "ols-target",
        )
        self.assertEqual(combined["ols-bootstrap"]["version"], "1.0")
        self.assertNotIn("ols-bootstrap", runtime)

    def test_explicit_incompatibility_preserves_generated_pins(self):
        """Fail on a real constraint conflict, without publishing partial output."""
        project = Path("pyproject.toml")
        project.write_text(
            project.read_text(encoding="utf-8").replace(
                '"ols-unrelated"]', '"ols-unrelated", "ols-dependency<2"]'
            ),
            encoding="utf-8",
        )
        pins = Path(".konflux/requirements.hashes.wheel.txt").read_bytes()
        with self.assertRaisesRegex(RuntimeError, "No solution"):
            resolver.uv_resolve(
                "3.12", "file:///io/primary/", "", upgrade_package="ols-target"
            )
        self.assertEqual(
            Path(".konflux/requirements.hashes.wheel.txt").read_bytes(), pins
        )


if __name__ == "__main__":
    unittest.main()
