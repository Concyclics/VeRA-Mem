"""Check English presentation, local document links, and localization provenance.

Standard library only. Does not access the network, load a model, or change files.
Use --verify-originals to compare localized files with the pinned local Git commit.
Historical model-run validators remain separate and retain their strict hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0003134f]")
LINK = re.compile(r"\[[^\]\n]*\]\(([^)\n]+)\)")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def repository_files(root):
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root, capture_output=True, check=True,
    )
    return sorted({p for p in result.stdout.decode().split("\0") if p})


def json_changes(before, after, path=""):
    """Allow translated strings only; retain exact structure and scalar types."""
    if type(before) is not type(after):
        raise ValueError(f"JSON type changed at {path}")
    if isinstance(before, dict):
        if before.keys() != after.keys():
            raise ValueError(f"JSON keys changed at {path}")
        return [item for key in before for item in json_changes(
            before[key], after[key], path + "/" + key.replace("~", "~0").replace("/", "~1"))]
    if isinstance(before, list):
        if len(before) != len(after):
            raise ValueError(f"JSON array length changed at {path}")
        return [item for i, (a, b) in enumerate(zip(before, after))
                for item in json_changes(a, b, path + "/" + str(i))]
    if before == after:
        return []
    if isinstance(before, str) and HAN.search(before) and not HAN.search(after):
        return [path]
    raise ValueError(f"Non-translation JSON change at {path}")


def headings(text):
    anchors = set(re.findall(r'<a\s+(?:id|name)=[\'"]([^\'"]+)', text))
    counts = {}
    for line in text.splitlines():
        match = re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", line)
        if not match:
            continue
        slug = re.sub(r"<[^>]*>", "", match.group(1)).lower()
        slug = re.sub(r"[^\w\-\s]", "", slug).replace(" ", "-")
        count = counts.get(slug, 0)
        counts[slug] = count + 1
        anchors.add(slug if count == 0 else slug + "-" + str(count))
    return anchors


def check(root=ROOT, verify_originals=False):
    files = repository_files(root)
    errors = []
    texts = {}
    json_files = 0
    for name in files:
        path = root / name
        if not path.is_file():
            errors.append(f"Missing tracked file: {name}")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        texts[name] = text
        if HAN.search(text):
            errors.append(f"Non-English Han text: {name}")
        if path.suffix == ".json":
            json_files += 1
            try:
                value = json.loads(text)
                if HAN.search(json.dumps(value, ensure_ascii=False)):
                    errors.append(f"Non-English decoded JSON text: {name}")
            except ValueError as error:
                errors.append(f"Invalid JSON {name}: {error}")
    links = 0
    for name, text in texts.items():
        if not name.endswith(".md"):
            continue
        # Ignore illustrative links inside fenced code blocks.
        prose = re.sub(r"```.*?```", "", text, flags=re.S)
        for match in LINK.finditer(prose):
            target = match.group(1).strip().split(' "', 1)[0].strip("<>")
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc:
                continue
            links += 1
            path = (root / name).parent / unquote(parsed.path) if parsed.path else root / name
            path = path.resolve()
            if not path.exists():
                errors.append(f"Broken local link in {name}: {target}")
                continue
            if parsed.fragment and path.suffix == ".md":
                fragment = unquote(parsed.fragment)
                if fragment not in headings(path.read_text()):
                    errors.append(f"Broken local anchor in {name}: {target}")
    manifest_path = root / "docs/localization_manifest.json"
    entries = []
    verified = 0
    try:
        manifest = json.loads(manifest_path.read_text())
        entries = manifest["files"]
        for entry in entries + manifest["unchanged_runtime"] + manifest["unchanged_sealed_artifacts"]:
            data = (root / entry["path"]).read_bytes()
            if sha(data) != entry["current_sha256"]:
                errors.append(f"Current localization/provenance SHA differs: {entry['path']}")
            if verify_originals:
                result = subprocess.run(["git", "show", manifest["original_commit"] + ":" + entry["path"]],
                                        cwd=root, capture_output=True, check=True)
                if sha(result.stdout) != entry["original_sha256"]:
                    errors.append(f"Original Git SHA differs: {entry['path']}")
                if entry.get("json_translated_paths") is not None:
                    changed = json_changes(json.loads(result.stdout), json.loads(data))
                    if changed != entry["json_translated_paths"]:
                        errors.append(f"Translated JSON paths differ: {entry['path']}")
                verified += 1
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        errors.append(f"Localization provenance check failed: {error}")
    return dict(passed=not errors, files_checked=len(files), text_files_checked=len(texts),
                json_files_checked=json_files, local_links_checked=links,
                localized_files=len(entries), original_git_blobs_verified=verified,
                errors=errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-originals", action="store_true")
    args = parser.parse_args()
    result = check(verify_originals=args.verify_originals)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
