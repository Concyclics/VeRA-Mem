"""Article-preserving WikiText preparation for the independently versioned corpus study.

Raw documents and token arrays belong outside the public source repository.
No test document is part of the training memory, tokenizer fit, or statistics fit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np

REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
SPLITS = ("train", "validation", "test")
TITLE = re.compile(r"^\s*=\s+([^=].*?)\s+=\s*$")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for piece in iter(lambda: stream.read(8 << 20), b""):
            digest.update(piece)
    return digest.hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=True) + "\n")
    temporary.replace(path)


def articles(rows):
    """Keep headings, subheadings, blank lines, and the boundary across shards."""
    title, lines = None, []
    iterator = iter(rows)
    row = next(iterator, None)
    previous_blank = True
    while row is not None:
        following = next(iterator, None)
        match = TITLE.fullmatch(row)
        if (match and not match.group(1).rstrip().endswith("=") and previous_blank
                and (following is None or not following.strip())):
            if title is not None:
                yield title, "".join(lines)
            elif any(line.strip() for line in lines):
                raise ValueError("Nonempty text precedes the first article heading")
            title, lines = match.group(1).strip(), []
        if title is not None:
            lines.append(row if row.endswith("\n") else row + "\n")
        previous_blank = not row.strip()
        row = following
    if title is not None:
        yield title, "".join(lines)


def windows(length, size=1024, stride=1024):
    """(input start, exclusive end, first scored target offset).

    Input contains a BOS and targets contain an EOS. Every next-token target
    occurs exactly once. Evaluation uses stride 512; initial/tail windows can
    score fewer than 512 tokens. Context never crosses an article boundary.
    """
    if not 1 <= stride <= size:
        raise ValueError("Require 1 <= stride <= size")
    for first_target in range(1, length, stride):
        end = min(length, first_target + stride)
        start = max(0, end - size - 1)
        yield start, end, first_target - start


class Corpus:
    def __init__(self, directory, split):
        self.root = Path(directory)
        self.split = split
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.tokens = np.load(self.root / f"{split}.tokens.npy", mmap_mode="r")
        self.offsets = np.load(self.root / f"{split}.offsets.npy", mmap_mode="r")
        self.chunks = np.load(self.root / f"{split}.chunks.npy", mmap_mode="r")
        groups = self.root / f"{split}.article_groups.npy"
        self.article_groups = (np.load(groups, mmap_mode="r") if groups.exists()
                               else np.arange(len(self.offsets) - 1, dtype=np.int64))

    def document(self, index):
        return self.tokens[self.offsets[index]:self.offsets[index + 1]]

    def chunk(self, index):
        doc, start, end = self.chunks[index]
        offset = self.offsets[doc]
        return self.tokens[offset + start:offset + end]

    def window_table(self, size=1024, stride=1024):
        return np.asarray([(doc, start, end, first)
                           for doc in range(len(self.offsets) - 1)
                           for start, end, first in windows(
                               int(self.offsets[doc + 1] - self.offsets[doc]), size, stride)],
                          dtype=np.int64)

    def batch(self, records, pad):
        length = max(int(r[2] - r[1]) for r in records)
        inputs = np.full((len(records), length - 1), pad, dtype=np.int64)
        targets = np.full_like(inputs, -100)
        mask = np.zeros_like(inputs)
        for i, (doc, start, end, first) in enumerate(records):
            sequence = self.document(doc)[start:end]
            inputs[i, :len(sequence) - 1] = sequence[:-1]
            mask[i, :len(sequence) - 1] = 1
            targets[i, first - 1:len(sequence) - 1] = sequence[first:]
        return inputs, targets, mask


def prepare(raw, tokenizer_path, output):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    output = Path(output)
    if (output / "manifest.json").exists():
        raise FileExistsError("Prepared corpus already exists; use a new destination")
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    boundary = tokenizer.eos_token_id
    if boundary is None:
        raise ValueError("Tokenizer must define EOS; it is also the document BOS")
    manifest = {"schema": "vera-corpus-data-v3", "dataset": "Salesforce/wikitext",
                "revision": REVISION, "configuration": "wikitext-103-raw-v1",
                "boundary_token_id": boundary, "pad_token_id": tokenizer.pad_token_id or boundary,
                "chunk_tokens": 128, "slots": 8,
                "text_rule": "Preserve rows, add newline to empty/nonterminated rows; retain headings",
                "article_boundary_rule": "Single-equals outer heading and blank rows on both sides; internal equals in titles are retained",
                "tokenizer_files": {p.name: sha256(p) for p in sorted(Path(tokenizer_path).iterdir())
                                    if p.is_file()}, "splits": {}, "raw_files": {},
                "cross_split_overlaps": []}
    seen_titles, seen_text = {}, {}
    for split in SPLITS:
        began = time.monotonic()
        files = sorted(Path(raw).glob(f"wikitext-103-raw-v1/{split}-*.parquet"))
        if not files:
            raise FileNotFoundError(split)
        for path in files:
            manifest["raw_files"][path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        def rows():
            for path in files:
                for batch in pq.ParquetFile(path).iter_batches(batch_size=16384, columns=["text"]):
                    yield from batch.column(0).to_pylist()
        offsets, chunks, token_count, documents = [0], [], 0, 0
        flat_path = output / f"{split}.tokens.bin"
        with flat_path.open("wb") as flat, (output / f"{split}.documents.jsonl").open("w") as metadata:
            pending = []
            def consume(pending):
                nonlocal token_count, documents
                encoded = tokenizer([text for _, text in pending], add_special_tokens=False,
                                    truncation=False, return_attention_mask=False)["input_ids"]
                for (title, text), ids in zip(pending, encoded):
                    text_hash = hashlib.sha256(text.encode()).hexdigest()
                    canonical_hash = hashlib.sha256(text.strip().encode()).hexdigest()
                    title_key = title.casefold()
                    overlaps = []
                    for kind, seen, key in (("title", seen_titles, title_key),
                                            ("stripped_text", seen_text, canonical_hash)):
                        if key in seen and seen[key][0] != split:
                            other_split, other_id = seen[key]
                            overlaps.append({"kind": kind, "other_split": other_split, "other_id": other_id})
                            manifest["cross_split_overlaps"].append(
                                {"split": split, "id": documents, "title": title, **overlaps[-1]})
                        else:
                            seen[key] = (split, documents)
                    sequence = np.asarray([boundary, *ids, boundary], dtype=np.int32)
                    flat.write(sequence.tobytes())
                    token_count += len(sequence)
                    offsets.append(token_count)
                    for start in range(1, len(sequence) - 1, 128):
                        chunks.append((documents, start, min(start + 128, len(sequence) - 1)))
                    metadata.write(json.dumps({"id": documents, "title": title,
                                               "text_sha256": text_hash,
                                               "text": text, "tokens": len(ids),
                                               "cross_split_overlaps": overlaps}, ensure_ascii=True) + "\n")
                    documents += 1
            for article in articles(rows()):
                pending.append(article)
                if len(pending) == 32:
                    consume(pending)
                    pending = []
            if pending:
                consume(pending)
        source = np.memmap(flat_path, dtype=np.int32, mode="r", shape=(token_count,))
        destination = np.lib.format.open_memmap(output / f"{split}.tokens.npy", mode="w+",
                                               dtype=np.int32, shape=source.shape)
        destination[:] = source
        destination.flush()
        del destination, source
        flat_path.unlink()
        np.save(output / f"{split}.offsets.npy", np.asarray(offsets, dtype=np.int64))
        np.save(output / f"{split}.chunks.npy", np.asarray(chunks, dtype=np.int64))
        # Group exact/canonical-title duplicate articles for strict retrieval
        # exclusion without discarding any official training tokens.
        parent = np.arange(documents, dtype=np.int64)
        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return int(i)
        title_groups, text_groups = {}, {}
        with (output / f"{split}.documents.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                i = row["id"]
                for lookup, key in ((title_groups, row["title"].casefold()),
                                    (text_groups, hashlib.sha256(row["text"].strip().encode()).hexdigest())):
                    if key in lookup:
                        a, b = root(i), root(lookup[key])
                        parent[max(a, b)] = min(a, b)
                    else:
                        lookup[key] = i
        groups = np.asarray([root(i) for i in range(documents)], dtype=np.int64)
        np.save(output / f"{split}.article_groups.npy", groups)
        manifest["splits"][split] = {"articles": documents, "text_tokens": token_count - 2 * documents,
                                    "lm_target_tokens": token_count - documents, "chunks": len(chunks),
                                    "canonical_article_groups": len(set(groups.tolist())),
                                    "duplicate_article_instances": documents - len(set(groups.tolist())),
                                    "seconds": time.monotonic() - began}
        print(json.dumps({"split": split, **manifest["splits"][split]}), flush=True)
    manifest["files"] = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
                         for p in sorted(output.iterdir()) if p.is_file()}
    manifest["confirmation_rule"] = "Exclude held-out articles overlapping an earlier split by title or stripped text"
    atomic_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prepare(args.raw, args.tokenizer, args.output)
