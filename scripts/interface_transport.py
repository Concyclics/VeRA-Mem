"""Shared private SSH multiplexing for local interface experiment tools.

Only standard-library filesystem operations and command construction occur.
Nothing connects to a host until the caller executes a returned command.
The controller, launcher and backup process must use the same workspace path;
its hash selects a short /tmp/vi-UID-HASH private directory, not a symlink.
No ~/.ssh/config changes or private credentials are required.

Examples::

    subprocess.run(ssh_command("xtrah100") + ["python3", "-"], input=code, ...)
    subprocess.run(["rsync", "-az", "-e", rsync_transport(), source, target], ...)
    subprocess.run(control_check_command("xtrah100"), timeout=20, ...)

``control_check_command`` checks an existing local master without opening a
new remote connection. A nonzero result can simply mean no master exists yet.
``connection_probe_command`` builds a no-output remote ``true`` probe: when
executed, it reuses an existing master or creates one that persists 120 seconds.
The helpers never close a master shared by another process.
"""
from __future__ import annotations

import os
import hashlib
from pathlib import Path
import re
import shlex
import stat


DEFAULT_WORKSPACE = Path(__file__).resolve().parents[2]


def _control_directory(workspace=None) -> Path:
    base = Path(DEFAULT_WORKSPACE if workspace is None else workspace).resolve(strict=True)
    if not base.is_dir():
        raise ValueError("SSH workspace must be an existing directory")
    workspace_hash = hashlib.sha256(os.fsencode(str(base))).hexdigest()[:8]
    # OpenSSH binds a temporary name with a further 17-character suffix. Deep
    # project paths cannot safely hold a %C socket even if the final name fits.
    return Path("/tmp") / f"vi-{os.geteuid()}-{workspace_hash}"


def control_path(workspace=None) -> str:
    """Return an OpenSSH ControlPath after securing its private directory.

    Reject symlink directories, foreign ownership, and unsafe existing children
    before using the directory. A directory FD and O_NOFOLLOW protect the final
    component while permissions are set. Only the current effective UID may
    own the directory/socket entries. Existing owner-controlled permissions are
    tightened to 0700. ``%C`` is OpenSSH's connection-specific hash expansion.
    """
    directory = _control_directory(workspace)
    try:
        directory.mkdir(mode=0o700)
    except FileExistsError:
        pass
    info = directory.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError("SSH control directory must be a real directory, never a symlink")
    if info.st_uid != os.geteuid():
        raise PermissionError("SSH control directory belongs to another user")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(directory, flags)
    try:
        opened = os.fstat(descriptor)
        if opened.st_uid != os.geteuid() or (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
            raise PermissionError("SSH control directory changed identity or owner")
        os.fchmod(descriptor, 0o700)
        # This also prevents a stale, previously world-writable directory from
        # supplying somebody else's socket or a symlink to another location.
        for name in os.listdir(descriptor):
            try:
                entry = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                # Another same-user SSH master may atomically rename/unlink
                # its temporary socket while these shared helpers inspect it.
                continue
            if stat.S_ISLNK(entry.st_mode):
                raise ValueError("SSH control directory contains a symlink")
            if entry.st_uid != os.geteuid():
                raise PermissionError("SSH control directory contains another user's entry")
        current = directory.lstat()
        if stat.S_ISLNK(current.st_mode) or (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino):
            raise PermissionError("SSH control directory was replaced")
    finally:
        os.close(descriptor)
    # %C expands to 40 bytes; OpenSSH's bind name appends '.' and 16 random
    # bytes. Both must fit before the NUL in Linux's 108-byte sockaddr_un.
    expanded = str(directory / ("cm-" + "0" * 40 + "." + "0" * 16))
    if len(os.fsencode(expanded)) >= 108:
        raise ValueError("SSH temporary control socket path would exceed sockaddr_un")
    return str(directory) + "/cm-%C"


def _host(host):
    if not isinstance(host, str) or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@:-]*", host):
        raise ValueError("Use a plain SSH alias or user@hostname, not command-line options")
    return host


def ssh_command(host=None, *, workspace=None) -> list[str]:
    """Return the reusable SSH prefix, optionally including a validated host."""
    if host is not None:
        _host(host)
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ControlMaster=auto",
               "-o", "ControlPersist=120", "-o", "ControlPath=" + control_path(workspace),
               "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15",
               "-o", "ServerAliveCountMax=3"]
    return command + ([host] if host is not None else [])


def rsync_transport(workspace=None) -> str:
    """Shell-quoted transport suitable for one rsync ``-e`` argument."""
    return shlex.join(ssh_command(workspace=workspace))


def control_check_command(host, *, workspace=None) -> list[str]:
    """Construct ``ssh -O check``; caller executes it to inspect local reuse."""
    return ssh_command(workspace=workspace) + ["-O", "check", _host(host)]


def connection_probe_command(host, *, workspace=None) -> list[str]:
    """Construct a reusable, remote read-only probe; do not execute it here."""
    return ssh_command(host, workspace=workspace) + ["true"]


def self_test():
    """CPU/filesystem-only tests. No SSH or rsync process is ever started."""
    import tempfile
    import shutil
    import unittest
    from unittest.mock import patch

    class Checks(unittest.TestCase):
        def directory(self, workspace):
            directory = _control_directory(workspace)
            self.addCleanup(lambda: shutil.rmtree(directory, ignore_errors=True) if not directory.is_symlink() else directory.unlink())
            return directory

        def test_same_private_socket_and_required_options_for_all_commands(self):
            with tempfile.TemporaryDirectory(prefix="it-") as temporary:
                workspace = Path(temporary)
                directory = self.directory(workspace)
                path = control_path(workspace)
                self.assertEqual(path, control_path(workspace))
                self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o700)
                command = ssh_command("xtrah100", workspace=workspace)
                for option in ("ControlMaster=auto", "ControlPersist=120", "ConnectTimeout=15",
                               "ServerAliveInterval=15", "ServerAliveCountMax=3", "ControlPath=" + path):
                    self.assertIn(option, command)
                self.assertEqual(shlex.split(rsync_transport(workspace)), command[:-1])
                self.assertEqual(control_check_command("xtrah100", workspace=workspace)[-3:], ["-O", "check", "xtrah100"])
                self.assertEqual(connection_probe_command("xtrah100", workspace=workspace), command + ["true"])

        def test_owner_directory_permissions_are_tightened(self):
            with tempfile.TemporaryDirectory(prefix="it-") as temporary:
                directory = self.directory(temporary)
                directory.mkdir(mode=0o755)
                control_path(temporary)
                self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o700)

        def test_directory_and_socket_symlinks_are_rejected(self):
            with tempfile.TemporaryDirectory(prefix="it-") as temporary:
                workspace = Path(temporary)
                target = workspace / "target"
                target.mkdir()
                directory = self.directory(workspace)
                directory.symlink_to(target, target_is_directory=True)
                with self.assertRaisesRegex(ValueError, "symlink"):
                    control_path(workspace)
                directory.unlink()
                directory.mkdir(mode=0o700)
                (directory / "cm-malicious").symlink_to(target)
                with self.assertRaisesRegex(ValueError, "symlink"):
                    control_path(workspace)

        def test_foreign_ownership_is_refused_without_chmod(self):
            with tempfile.TemporaryDirectory(prefix="it-") as temporary:
                directory = self.directory(temporary)
                directory.mkdir(mode=0o755)
                with patch(__name__ + "._control_directory", return_value=directory), patch("os.geteuid", return_value=os.geteuid() + 1):
                    with self.assertRaises(PermissionError):
                        control_path(temporary)
                self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o755)

        def test_spaces_are_quoted_and_bad_hosts_cannot_become_options(self):
            with tempfile.TemporaryDirectory(prefix="it-") as temporary:
                workspace = Path(temporary) / "space here"
                workspace.mkdir()
                self.directory(workspace)
                self.assertEqual(shlex.split(rsync_transport(workspace)), ssh_command(workspace=workspace))
                for host in ("-Oexit", "xtrah100\ncommand", "host command", "", None):
                    with self.assertRaises(ValueError):
                        control_check_command(host, workspace=workspace)

        def test_real_workspace_and_internal_ssh_suffix_fit_linux_socket_limit(self):
            # Inspect the real path computation without creating/removing its
            # live directory or starting an SSH connection.
            directory = _control_directory(DEFAULT_WORKSPACE)
            self.assertEqual(directory.parent, Path("/tmp"))
            self.assertRegex(directory.name, r"^vi-[0-9]+-[0-9a-f]{8}$")
            expanded = str(directory / ("cm-" + "0" * 40 + "." + "0" * 16))
            self.assertLess(len(os.fsencode(expanded)), 108)

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    import sys
    if sys.argv[1:] != ["--self-test"]:
        raise SystemExit("Import these command builders, or run --self-test (never opens a connection).")
    raise SystemExit(self_test())
