"""Validate paired coast/maneuver scenario manifests before evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import jsonschema
import yaml


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def validate_manifest(manifest_path: Path, schema_path: Path, require_runnable: bool) -> dict:
    manifest = load_yaml(manifest_path)
    schema = load_yaml(schema_path)
    jsonschema.Draft202012Validator(schema).validate(manifest)

    if require_runnable:
        if not manifest["runnable"]:
            raise ValueError("Manifest is schema-valid but is not marked runnable")
        if manifest["initial_state_snapshot"]["status"] != "ready":
            raise ValueError("Runnable manifest requires a ready initial-state snapshot")
        if manifest["maneuver_plan"]["status"] != "ready":
            raise ValueError("Runnable manifest requires a calibrated maneuver plan")
        if not manifest["maneuver_plan"]["events"]:
            raise ValueError("Runnable paired maneuver manifest requires at least one maneuver event")

    return manifest


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "manifest",
        type=Path,
        nargs="?",
        default=project_root / "config" / "scenario_manifest.example.yaml",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=project_root / "config" / "scenario_manifest.schema.yaml",
    )
    parser.add_argument("--require-runnable", action="store_true")
    args = parser.parse_args()

    manifest = validate_manifest(args.manifest, args.schema, args.require_runnable)
    state = "runnable" if manifest["runnable"] else "not runnable (initial snapshot/events pending)"
    print(f"VALID {manifest['manifest_id']}: {state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
