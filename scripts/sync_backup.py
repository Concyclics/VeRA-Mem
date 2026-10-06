"""Incrementally back up experiment artifacts without public base-model weights.

Uses SSH configuration on the caller's machine; no credentials are stored here.
No deletion flag is used. --verify runs checksum dry-runs after synchronization.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
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
p.add_argument("--ssh-control-path",type=Path,help="Optional caller-owned SSH control socket; remains open for reuse")
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
manifest = {"host_alias": a.host,
    "records": records, "base_model_weights_copied": False,
    "complete": False, "checksum_verified": False,
    "note": "An active run can change during sync. Final verification requires completed jobs."}


def save_manifest():
    manifest["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    temporary = local / "backup_manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2)+"\n")
    temporary.replace(local / "backup_manifest.json")


save_manifest()
with tempfile.TemporaryDirectory(prefix="vera_mem_sync_") as socket_directory:
    # Reuse one authenticated transport, avoiding bursts of SSH connections.
    # The private temporary directory prevents socket-name collisions.
    if a.ssh_control_path is not None:
        if not a.ssh_control_path.is_absolute(): raise ValueError("Control path must be absolute")
        import os
        parent=a.ssh_control_path.parent
        if parent.is_symlink() or parent.stat().st_uid!=os.getuid() or parent.stat().st_mode & 0o077:
            raise ValueError("Caller control socket must be in an owned private directory")
    control = str(a.ssh_control_path or Path(socket_directory) / "connection")
    transport = shlex.join(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
        "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
        "-o", "ControlMaster=auto", "-o", "ControlPersist=120", "-o", "ControlPath=" + control])
    common = ["--protect-args", "-e", transport, "--exclude=.git", "--exclude=__pycache__", "--exclude=.pytest_cache"]
    try:
        for source, destination in plans:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.endswith("/"):
                destination.mkdir(parents=True, exist_ok=True)
            remote = f"{a.host}:{a.remote_root.rstrip('/')}/{source}"
            retry_run(["rsync", "-az", *common, remote, str(destination)])
            record = {"source": source, "destination": str(destination)}
            if a.verify:
                result = retry_run(["rsync", "-rcn", "--itemize-changes", *common, remote, str(destination)], text=True, capture_output=True)
                record["checksum_differences"] = result.stdout.splitlines()
            records.append(record)
            save_manifest()
        manifest["complete"] = True
        manifest["checksum_verified"] = a.verify and all(not r.get("checksum_differences") for r in records)
    except Exception as error:
        manifest["error"] = str(error)
        raise
    finally:
        save_manifest()
        if a.ssh_control_path is None:
            subprocess.run(["ssh", "-S", control, "-O", "exit", a.host],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
print(json.dumps(manifest, indent=2))
