from copy import deepcopy
from pathlib import Path
import unittest

import yaml

from real_environment.scenario_manifest import validate_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = PROJECT_ROOT / "config" / "scenario_manifest.schema.yaml"
EXAMPLE_PATH = PROJECT_ROOT / "config" / "scenario_manifest.example.yaml"


class ScenarioManifestTests(unittest.TestCase):
    def test_example_is_schema_valid_but_not_runnable(self):
        manifest = validate_manifest(EXAMPLE_PATH, SCHEMA_PATH, require_runnable=False)
        self.assertFalse(manifest["runnable"])

        with self.assertRaisesRegex(ValueError, "not marked runnable"):
            validate_manifest(EXAMPLE_PATH, SCHEMA_PATH, require_runnable=True)

    def test_ready_manifest_requires_at_least_one_maneuver_event(self):
        manifest = yaml.safe_load(EXAMPLE_PATH.read_text(encoding="utf-8"))
        runnable = deepcopy(manifest)
        runnable["runnable"] = True
        runnable["initial_state_snapshot"].update(
            status="ready",
            path="results/scenarios/seed_000_initial_state.npz",
            sha256="0" * 64,
        )
        runnable["maneuver_plan"]["status"] = "ready"

        temporary_manifest = PROJECT_ROOT / "results" / "test_manifest_no_events.yaml"
        temporary_manifest.parent.mkdir(parents=True, exist_ok=True)
        temporary_manifest.write_text(yaml.safe_dump(runnable, sort_keys=False), encoding="utf-8")
        self.addCleanup(temporary_manifest.unlink, missing_ok=True)

        with self.assertRaisesRegex(ValueError, "at least one maneuver event"):
            validate_manifest(temporary_manifest, SCHEMA_PATH, require_runnable=True)


if __name__ == "__main__":
    unittest.main()
