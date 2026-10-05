"""Deterministic, auditable datasets for small memory experiments.

The synthetic path and JSONL helpers require only the Python standard library.
MedMCQA preparation downloads only pinned train/validation parquet files and
requires an installed parquet reader (pyarrow, pandas, or datasets).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen


@dataclass
class Example:
    id: str
    question: str
    answer: str
    support: str
    paraphrase: str
    choices: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


MEMORY_WORDS = (
    "apple", "river", "candle", "garden", "tiger", "pencil", "window", "basket",
    "cloud", "lemon", "mirror", "rabbit", "island", "silver", "orange", "forest",
)
MEDMCQA_REPO = "openlifescienceai/medmcqa"


def _check_counts(n_train: int, n_control: int) -> None:
    for name, count in (("n_train", n_train), ("n_control", n_control)):
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"{name} must be a nonnegative integer")


def synthetic_dataset(
    seed: int = 42, n_train: int = 32, n_control: int = 16
) -> dict[str, list[Example]]:
    """Generate random associations; controls are never written by the runner.

    Values are balanced to within one occurrence within each split. Entity IDs
    have no deterministic semantic or ordinal relationship to their values.
    The control labels are hidden random values, not abstention targets; their
    accuracy measures unexposed guessing, not an abstention score.
    """
    _check_counts(n_train, n_control)
    rng = random.Random(seed)
    used_entities: set[str] = set()
    result: dict[str, list[Example]] = {}
    for split, count in (("stream", n_train), ("control", n_control)):
        # Shuffle the vocabulary before repeating to avoid favoring its prefix.
        vocabulary = list(MEMORY_WORDS)
        rng.shuffle(vocabulary)
        answers = [vocabulary[i % len(vocabulary)] for i in range(count)]
        rng.shuffle(answers)
        examples = []
        for answer in answers:
            entity = "N" + f"{rng.getrandbits(64):016x}"
            while entity in used_entities:
                entity = "N" + f"{rng.getrandbits(64):016x}"
            used_entities.add(entity)
            examples.append(Example(
                id=f"synthetic-{entity}",
                question=(f"What is the assigned memory word for entity {entity}? "
                          "Reply with only the word."),
                answer=answer,
                support=f"The assigned memory word for entity {entity} is {answer}.",
                paraphrase=(f"Recall the memory word associated with {entity}. "
                            "Return only that word."),
                metadata={"dataset": "synthetic-associations-v1", "seed": seed,
                          "entity": entity, "split": split,
                          "never_written": split == "control"},
            ))
        result[split] = examples
    return result


def _read_json_url(url: str) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": "vera-mem-research/0.1"})
    with urlopen(request, timeout=120) as response:
        return json.load(response)


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    request = Request(url, headers={"User-Agent": "vera-mem-research/0.1"})
    try:
        with urlopen(request, timeout=120) as response, partial.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        partial.replace(destination)
    finally:
        if partial.exists():
            partial.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_parquet_rows(paths: list[Path]) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        pass
    else:
        result = []
        for path in paths:
            result.extend(pq.read_table(path).to_pylist())
        return result
    try:
        import pandas as pd
        return [row for path in paths
                for row in pd.read_parquet(path).to_dict(orient="records")]
    except ImportError:
        pass
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise ImportError(
            "MedMCQA preparation needs a parquet reader. Install pyarrow, or "
            "pandas with a parquet engine, or datasets. Synthetic data does not."
        ) from error
    return list(load_dataset("parquet", data_files=[str(path) for path in paths],
                             split="train"))


def _question_hash(row: dict[str, Any]) -> str:
    """Case/whitespace normalized question and options, independent of source ID."""
    parts = [str(row[name]) for name in ("question", "opa", "opb", "opc", "opd")]
    canonical = "\n".join(" ".join(part.casefold().split()) for part in parts)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _medmcqa_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    seen: set[str] = set()
    for row in rows:
        if row.get("choice_type") != "single":
            continue
        if any(not isinstance(row.get(name), str) or not row[name].strip()
               for name in ("question", "opa", "opb", "opc", "opd")):
            continue
        # The official HF parquet schema is a zero-based four-class label.
        # Reject invalid values rather than silently applying an off-by-one fix.
        label = row.get("cop")
        if isinstance(label, bool) or not isinstance(label, int) or label not in range(4):
            raise ValueError(f"Expected MedMCQA cop in 0..3, got {label!r}")
        fingerprint = _question_hash(row)
        if fingerprint not in seen:
            result.append(row)
            seen.add(fingerprint)
    return result


def _medmcqa_example(row: dict[str, Any], split: str) -> Example:
    options = [row[f"op{letter}"].strip() for letter in "abcd"]
    choice_text = "\n".join(f"{letter}. {option}"
                            for letter, option in zip("ABCD", options))
    question = row["question"].strip()
    index = row["cop"]
    letter = "ABCD"[index]
    fingerprint = _question_hash(row)
    source_id = str(row.get("id", fingerprint))
    return Example(
        id=f"medmcqa-{split}-{source_id}",
        question=(f"Question: {question}\nOptions:\n{choice_text}\n"
                  "Answer with only the correct option letter (A, B, C, or D)."),
        answer=letter,
        support=(f"Question: {question}\nOptions:\n{choice_text}\n"
                 f"The correct answer is {letter}. {options[index]}"),
        paraphrase=(f"Select the correct option for this question:\n{question}\n"
                    f"{choice_text}\nReturn only its letter (A, B, C, or D)."),
        choices=options,
        metadata={"dataset": MEDMCQA_REPO, "source_split": split,
                  "source_id": source_id, "question_sha256": fingerprint,
                  "correct_option_index": index,
                  "subject_name": row.get("subject_name"),
                  "prompt_rewording_only": True,
                  "never_written": split == "validation"},
    )


def prepare_medmcqa(
    output_dir: str | Path, seed: int = 42, n_train: int = 32, n_control: int = 16
) -> dict[str, list[Example]]:
    """Download pinned train/validation data and save examples plus provenance.

    A previous output manifest pins subsequent calls to the same revision.
    Existing source files must match the previous manifest checksums. A fresh
    directory resolves main exactly once through the HF API, then uses only
    immutable revision URLs. The benchmark test split is never downloaded.
    """
    from .metrics import save_examples

    _check_counts(n_train, n_control)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if previous is not None and previous.get("source_repo") != MEDMCQA_REPO:
        raise ValueError("Existing manifest does not describe the expected MedMCQA source")
    revision_suffix = f"/revision/{previous['source_revision']}" if previous else ""
    source = _read_json_url(f"https://huggingface.co/api/datasets/{MEDMCQA_REPO}{revision_suffix}")
    revision = source.get("sha", "")
    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", revision):
        raise ValueError("HF API did not return an immutable source commit SHA")
    if previous and revision != previous["source_revision"]:
        raise ValueError("HF API revision differs from the existing pinned manifest")
    old_hashes = {item["path"]: item["sha256"] for item in previous.get("source_files", [])} if previous else {}
    files = [item["rfilename"] for item in source.get("siblings", [])]
    source_files = []
    rows_by_split = {}
    for split in ("train", "validation"):
        names = sorted(name for name in files if re.search(
            rf"(?:^|/){split}(?:-|\.)[^/]*\.parquet$", name
        ))
        # Accept an unsharded train.parquet / validation.parquet as well.
        names += sorted(name for name in files if Path(name).name == f"{split}.parquet"
                        and name not in names)
        if not names:
            raise ValueError(f"No {split} parquet files found at source revision {revision}")
        paths = []
        for name in names:
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Unsafe source filename")
            destination = output / "source" / revision / name
            if not destination.exists():
                url = f"https://huggingface.co/datasets/{MEDMCQA_REPO}/resolve/{revision}/{quote(name, safe='/')}"
                _download_file(url, destination)
            digest = _sha256(destination)
            if name in old_hashes and old_hashes[name] != digest:
                raise ValueError(f"Cached source checksum mismatch: {name}")
            paths.append(destination)
            source_files.append({"path": name, "split": split, "sha256": digest,
                                 "bytes": destination.stat().st_size})
        rows_by_split[split] = _medmcqa_candidates(_read_parquet_rows(paths))

    rng = random.Random(seed)
    train = list(rows_by_split["train"])
    rng.shuffle(train)
    if len(train) < n_train:
        raise ValueError(f"Only {len(train)} eligible training examples for requested {n_train}")
    selected = train[:n_train]
    # Exclude overlap with the full eligible training pool, not only the sample.
    train_hashes = {_question_hash(row) for row in train}
    train_ids = {str(row["id"]) for row in train if "id" in row}
    controls = [row for row in rows_by_split["validation"]
                if _question_hash(row) not in train_hashes
                and str(row.get("id", "")) not in train_ids]
    rng.shuffle(controls)
    if len(controls) < n_control:
        raise ValueError(f"Only {len(controls)} disjoint validation examples for requested {n_control}")
    result = {"stream": [_medmcqa_example(row, "train") for row in selected],
              "control": [_medmcqa_example(row, "validation") for row in controls[:n_control]]}
    outputs = {}
    for split, examples in result.items():
        path = output / f"{split}.jsonl"
        save_examples(path, examples)
        outputs[split] = {"path": path.name, "count": len(examples),
                          "sha256": _sha256(path), "ids": [example.id for example in examples]}
    manifest = {"format_version": 1, "source_repo": MEDMCQA_REPO,
                "source_revision": revision, "source_files": source_files,
                "seed": seed, "outputs": outputs,
                "filters": ["choice_type=single", "nonempty question/options",
                            "deduplicate normalized question+options",
                            "validation disjoint from eligible training pool by ID and hash"],
                "label_schema": {"source_field": "cop", "source_values": [0, 1, 2, 3],
                                 "answer_values": ["A", "B", "C", "D"]},
                "excluded_fields": ["exp"],
                "paraphrase_scope": "prompt wrapper only; not semantic question paraphrases"}
    temporary = manifest_path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    return result
