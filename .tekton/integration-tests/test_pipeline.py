"""Offline regression checks for the Konflux service e2e pipeline."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).parent
PIPELINE = ROOT / "pipelines/lightspeed-service-e2e-test-pipeline.yaml"
SCENARIO = ROOT / "integration-test-scenarios/konflux-integration-test-scenarios.yaml"


class PipelineTests(unittest.TestCase):
    """Check snapshot behavior and manifest wiring without cluster credentials."""

    def setUp(self):
        """Load the pipeline under test."""
        self.assertTrue(PIPELINE.exists(), "Service HyperShift pipeline is missing")
        self.pipeline = yaml.safe_load(PIPELINE.read_text())
        self.tasks = {task["name"]: task for task in self.pipeline["spec"]["tasks"]}

    def extract(self, components):
        """Execute the real snapshot extraction script with Tekton result paths."""
        script = self.tasks["get-snapshot-component"]["taskSpec"]["steps"][0]["script"]
        with tempfile.TemporaryDirectory() as directory:
            for result in ("commit", "service-image"):
                script = script.replace(
                    f"$(step.results.{result}.path)", f"{directory}/{result}"
                )
            env = dict(
                os.environ,
                SNAPSHOT=json.dumps({"components": components}),
                COMPONENT_NAME="lightspeed-service",
            )
            process = subprocess.run(
                ["/bin/bash"], input=script, text=True, env=env, capture_output=True
            )
            values = {p.name: p.read_text() for p in Path(directory).iterdir()}
            return process.returncode, values

    def test_snapshot_selects_service_not_bundle(self):
        """Read the service revision and image even when a bundle appears first."""
        code, values = self.extract(
            [
                {
                    "name": "ols-bundle",
                    "containerImage": "wrong",
                    "source": {"git": {"revision": "bad"}},
                },
                {
                    "name": "lightspeed-service",
                    "containerImage": "quay.io/service@sha256:123",
                    "source": {"git": {"revision": "a" * 40}},
                },
            ]
        )
        self.assertEqual(code, 0)
        self.assertEqual(
            values, {"commit": "a" * 40, "service-image": "quay.io/service@sha256:123"}
        )

    def test_snapshot_rejects_invalid_components(self):
        """Fail early for absent, duplicate, empty, or malformed service metadata."""
        valid = {
            "name": "lightspeed-service",
            "containerImage": "quay.io/service:tag",
            "source": {"git": {"revision": "a" * 40}},
        }
        for components in (
            [],
            [valid, valid],
            [dict(valid, source={})],
            [dict(valid, containerImage=None)],
            [dict(valid, containerImage="")],
            [dict(valid, source={"git": {"revision": "main"}})],
        ):
            with self.subTest(components=components):
                self.assertNotEqual(self.extract(components)[0], 0)

    def test_pipeline_wires_snapshot_and_retains_failures(self):
        """Pass snapshot values into tests and preserve failures after publication."""
        task = self.tasks["ols-service-tests"]
        params = {param["name"]: param["value"] for param in task["params"]}
        self.assertEqual(
            params["commit"], "$(tasks.get-snapshot-component.results.commit)"
        )
        self.assertEqual(
            params["service-image"],
            "$(tasks.get-snapshot-component.results.service-image)",
        )
        steps = task["taskSpec"]["steps"]
        self.assertNotIn("gather-cluster-resources", [step["name"] for step in steps])
        run = steps[0]
        env = {value["name"]: value.get("value") for value in run["env"]}
        self.assertEqual(env["OLS_IMAGE"], "$(params.service-image)")
        self.assertIn(
            'git fetch --depth=1 origin "${SERVICE_COMMIT_SHA}"', run["script"]
        )
        self.assertIn("tests/scripts/test-e2e-cluster.sh", run["script"])
        self.assertNotIn("related_images", PIPELINE.read_text())
        self.assertEqual(run["onError"], "continue")
        self.assertEqual(steps[-1]["name"], "fail-if-any-step-failed")
        push = next(step for step in steps if step["name"] == "push-artifacts")
        push_params = {param["name"]: param["value"] for param in push["params"]}
        self.assertEqual(push_params["workdir-path"], "/workspace/artifacts")

    def test_hypershift_provisions_mirrors_and_disables_guest_mutations(self):
        """Provision the canonical mirrors through HostedCluster, not the guest API."""
        provision = self.tasks["provision-ephemeral-cluster"]
        params = {p["name"]: p["value"] for p in provision["params"]}
        provision_env = json.loads(params["env"])
        manifest = yaml.safe_load(
            (
                ROOT.parent.parent
                / "tests/config/operator_install/imagedigestmirrorset.yaml"
            ).read_text()
        )
        self.assertEqual(
            yaml.safe_load(provision_env.get("IMAGE_CONTENT_SOURCES", "")),
            manifest["spec"]["imageDigestMirrors"],
        )
        run = self.tasks["ols-service-tests"]["taskSpec"]["steps"][0]
        env = {e["name"]: e.get("value") for e in run["env"]}
        self.assertEqual(env.get("OLS_EXTERNAL_IMAGE_MIRRORS"), "true")

    def test_scenario_targets_service_pipeline(self):
        """Trigger for service builds and resolve an existing pipeline path."""
        scenario = yaml.safe_load(SCENARIO.read_text())
        self.assertEqual(
            scenario["spec"]["contexts"][0]["name"], "component_lightspeed-service"
        )
        params = {
            p["name"]: p["value"] for p in scenario["spec"]["resolverRef"]["params"]
        }
        self.assertEqual(
            params["url"], "https://github.com/openshift/lightspeed-service"
        )
        self.assertEqual(
            params["pathInRepo"], str(PIPELINE.relative_to(ROOT.parent.parent))
        )


if __name__ == "__main__":
    unittest.main()
