#!/usr/bin/env python3
"""Verify original encoder imports and explicitly recorded local adaptations."""
import hashlib
import json
from pathlib import Path
import tomllib


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "UPSTREAM.json").read_text())
    upstream = {entry["source"]: entry["sha256"] for entry in manifest["files"]}
    if len(upstream) != len(manifest["files"]):
        raise ValueError("Duplicate upstream source entry")
    local = {}
    path = root / "LOCAL_MODIFICATIONS.toml"
    if path.exists():
        modifications = tomllib.loads(path.read_text())
        if modifications["schema"] != 1 or modifications["upstream_revision"] != manifest["revision"]:
            raise ValueError("Local modifications refer to a different upstream revision")
        for entry in modifications["files"]:
            name = entry["source"]
            if name in local or name not in upstream:
                raise ValueError(f"Unknown or duplicate local modification: {name}")
            if entry["upstream_sha256"] != upstream[name]:
                raise ValueError(f"Local modification has wrong original hash: {name}")
            if not entry.get("changes") or not entry.get("verification"):
                raise ValueError(f"Local modification lacks change/verification record: {name}")
            local[name] = entry["local_sha256"]
    for name, original in upstream.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Source path escapes imported crate: {name}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != local.get(name, original):
            raise ValueError(f"Imported source changed outside its recorded hash: {name}")
    return dict(revision=manifest["revision"],files=len(upstream),verbatim=len(upstream)-len(local),adapted=len(local))


if __name__ == "__main__":
    try:
        result = verify(Path(__file__).resolve().parents[2] / "crates" / "burn_vjepa")
    except (ValueError, KeyError, OSError) as error:
        raise SystemExit(str(error)) from error
    print(f"Verified {result['files']} encoder files from {result['revision']}: "
          f"{result['verbatim']} verbatim, {result['adapted']} recorded adaptation(s)")
