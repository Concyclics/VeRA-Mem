"""Dependency-free serialization and conservative answer metrics."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import re
import unicodedata
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from .data import Example


def save_examples(path: str | Path, examples: Iterable[Example]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(asdict(example), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(destination)


def load_examples(path: str | Path) -> list[Example]:
    from .data import Example

    result = []
    ids = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                fields = json.loads(line)
                example = Example(**fields)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid example at {path}:{line_number}: {error}") from error
            for name in ("id", "question", "answer", "support", "paraphrase"):
                if not isinstance(getattr(example, name), str):
                    raise ValueError(f"Example {name} must be a string at {path}:{line_number}")
            if not isinstance(example.metadata, dict):
                raise ValueError(f"Example metadata must be a dictionary at {path}:{line_number}")
            if example.choices is not None and (
                not isinstance(example.choices, list)
                or any(not isinstance(choice, str) for choice in example.choices)
            ):
                raise ValueError(f"Example choices must be a string list at {path}:{line_number}")
            if example.id in ids:
                raise ValueError(f"Duplicate example ID at {path}:{line_number}: {example.id}")
            ids.add(example.id)
            result.append(example)
    return result


def normalize_answer(value: str) -> str:
    """Normalize case, Unicode punctuation and whitespace, preserving articles.

    Removing English articles would make the valid MCQ label 'A' empty.
    This is a declared local metric, not the official LongMemEval evaluator.
    """
    if not isinstance(value, str):
        raise TypeError("Answers must be strings")
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = "".join(" " if unicodedata.category(char).startswith("P") else char
                         for char in normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def exact_match(prediction: str, answer: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(answer))


def token_f1(prediction: str, answer: str) -> float:
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(answer).split()
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(predicted), overlap / len(expected)
    return 2 * precision * recall / (precision + recall)
