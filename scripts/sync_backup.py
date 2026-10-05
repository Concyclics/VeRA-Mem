"""Incrementally back up experiment artifacts without public base-model weights.

Uses SSH configuration on the caller's machine; no credentials are stored here.
No deletion flag is used. --verify runs checksum dry-runs after synchronization.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import time


def retry_run(command, **kwargs):
    for attempt in range(3):
        try:
            return subprocess.run(command, check=True, **kwargs)
        except subprocess.CalledProcessError:
            if attempt == 2:
                raise
            time.sleep(3 * (attempt + 1))

p = argparse.ArgumentParser()
p.add_argument("--host", required=True, help="Existing SSH alias")
p.add_argument("--remote-root", required=True)
p.add_argument("--local-root", type=Path, required=True)
p.add_argument("--verify", action="store_true")
a = p.parse_args()
if not re.fullmatch(r"[A-Za-z0-9_.@-]+", a.host) or a.host.startswith("-"):
    raise ValueError("Use a plain SSH alias or user@hostname")
if not a.remote_root.startswith("/") or "\n" in a.remote_root:
    raise ValueError("Remote root must be an absolute path")
local = a.local_root.resolve()
local.mkdir(parents=True, exist_ok=True)
plans = [
    ("runs/", local / "runs" / a.host),
    ("data/", local / "data" / a.host),
    ("VeRA-Mem/", local / "remote_snapshots" / a.host / "source"),
    ("models/manifest.json", local / "remote_snapshots" / a.host / "model_manifest.json"),
]
records = []
for source, destination in plans:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.endswith("/"):
        destination.mkdir(parents=True, exist_ok=True)
    remote = f"{a.host}:{a.remote_root.rstrip('/')}/{source}"
    cmd = ["rsync", "-az", "--protect-args", "--exclude=.git", "--exclude=__pycache__", "--exclude=.pytest_cache", remote, str(destination)]
    retry_run(cmd)
    record = {"source": source, "destination": str(destination)}
    if a.verify:
        verify = ["rsync", "-rcn", "--itemize-changes", "--protect-args", "--exclude=.git", "--exclude=__pycache__", "--exclude=.pytest_cache", remote, str(destination)]
        result = retry_run(verify, text=True, capture_output=True)
        record["checksum_differences"] = result.stdout.splitlines()
    records.append(record)
manifest = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "host_alias": a.host,
    "records": records, "base_model_weights_copied": False,
    "complete": True, "checksum_verified": a.verify and all(not r.get("checksum_differences") for r in records),
    "note": "An active run can change during sync. Final verification requires completed jobs."}
(local / "backup_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
print(json.dumps(manifest, indent=2))
