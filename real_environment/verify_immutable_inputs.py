"""Verify that the read-only Smith inputs match the recorded SHA-256 manifest."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
from pathlib import Path

import yaml


def find_repository_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "try_code" / "MARLSSA-main").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate the thesis-wiki repository root")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def excluded(relative_path: Path, patterns: list[str]) -> bool:
    relative = relative_path.as_posix()
    for pattern in patterns:
        if pattern == ".DS_Store" and relative_path.name == ".DS_Store":
            return True
        if fnmatch.fnmatch(relative, pattern) or fnmatch.fnmatch(f"./{relative}", pattern):
            return True
    return False


def sha256_tree(root: Path, exclusions: list[str]) -> str:
    aggregate = hashlib.sha256()
    relative_files = sorted(
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file() and not excluded(path.relative_to(root), exclusions)
    )
    for relative_path in relative_files:
        file_digest = sha256_file(root / relative_path)
        aggregate.update(f"{file_digest}  ./{relative_path.as_posix()}\n".encode("utf-8"))
    return aggregate.hexdigest()


def verify_manifest(manifest_path: Path) -> list[dict[str, str | bool]]:
    repository_root = find_repository_root(manifest_path.resolve())
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    checks: list[dict[str, str | bool]] = []

    for name, item in manifest["tree_inputs"].items():
        actual = sha256_tree(repository_root / item["path"], item.get("exclusions", []))
        checks.append(
            {
                "name": name,
                "kind": "tree",
                "expected": item["digest"],
                "actual": actual,
                "matches": actual == item["digest"],
            }
        )

    for name, item in manifest["file_inputs"].items():
        actual = sha256_file(repository_root / item["path"])
        checks.append(
            {
                "name": name,
                "kind": "file",
                "expected": item["digest"],
                "actual": actual,
                "matches": actual == item["digest"],
            }
        )

    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config" / "immutable_inputs.yaml",
    )
    args = parser.parse_args()

    checks = verify_manifest(args.manifest)
    for check in checks:
        status = "OK" if check["matches"] else "MISMATCH"
        print(f"{status:8} {check['kind']:4} {check['name']}")

    return 0 if all(bool(check["matches"]) for check in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
