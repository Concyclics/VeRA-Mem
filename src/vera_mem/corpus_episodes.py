"""Deterministic passage-completion episodes; these are not an open-domain QA benchmark.

The support is an observed natural WikiText chunk. Its independent query gives
an eight-word anchor and asks for the next three words. Controlled edits replace
that continuation in the observation itself. Selection never uses model success.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np

from .corpus_data import Corpus, atomic_json, sha256

WORD = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*")
TRIPLE = re.compile(r"\b[A-Za-z]{2,}(?: +[A-Za-z]{2,}){2}\b")
TRAIN_STYLES = (
    'In the remembered passage from "{title}", what are the next three words after "{anchor}"? Return only those words.',
    'Complete this excerpt from "{title}": {anchor} ___. Give exactly the three words in the stored passage.',
)
TEST_STYLES = (
    'Consult your memory of "{title}". The phrase "{anchor}" is immediately followed by which three words? Answer with the continuation alone.',
    'Recover the missing three-word continuation from the stored text titled "{title}".\nPrefix: {anchor}\nContinuation:',
)


def make_question(episode, style=0, heldout=False):
    return (TEST_STYLES if heldout else TRAIN_STYLES)[style].format(**episode)


def prompt_ids(tokenizer, episode, *, style=0, heldout=False, context=None):
    question = make_question(episode, style, heldout)
    if context is not None:
        question = "Remember this passage and answer using its exact wording:\n" + context + "\n\n" + question
    encoded = tokenizer.apply_chat_template([{"role": "user", "content": question}], tokenize=True,
                                            return_dict=False, add_generation_prompt=True, enable_thinking=False)
    # Transformers releases differ in their default chat-template return type.
    if isinstance(encoded, dict) or hasattr(encoded, "input_ids"):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError("A single episode unexpectedly produced a prompt batch")
        encoded = encoded[0]
    if not all(isinstance(token, int) for token in encoded):
        raise TypeError("Chat template did not return integer token IDs")
    return encoded


def answer_text(text):
    return " ".join(WORD.findall(text.casefold()))


def select(corpus, tokenizer, count, seed):
    metadata = {}
    excluded = set()
    with (corpus.root / f"{corpus.split}.documents.jsonl").open() as stream:
        for line in stream:
            doc = json.loads(line)
            metadata[doc["id"]] = doc["title"]
            if doc["cross_split_overlaps"]:
                excluded.add(doc["id"])
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(corpus.chunks))
    eligible = []
    for chunk in order:
        doc = int(corpus.chunks[chunk, 0])
        if doc in excluded:
            continue
        observed = corpus.chunk(chunk).tolist()
        if len(observed) < 64:
            continue
        support = tokenizer.decode(observed, skip_special_tokens=False)
        if "\ufffd" in support:
            continue
        candidates = [m for m in TRIPLE.finditer(support)
                      if .35 * len(support) < m.start() < .75 * len(support)
                      and support.count(m.group()) == 1]
        if not candidates:
            continue
        match = candidates[int(rng.integers(len(candidates)))]
        preceding = list(WORD.finditer(support[:match.start()]))
        if len(preceding) < 8:
            continue
        anchor = support[preceding[-8].start():match.start()].strip()
        row = {"id": f"{corpus.split}:{int(chunk)}", "chunk": int(chunk), "article": doc,
               "title": metadata[doc], "support": support, "support_ids": observed,
               "anchor": anchor, "answer": match.group(), "answer_start": match.start(),
               "answer_end": match.end()}
        if any(answer_text(row["answer"]) in answer_text(make_question(row, s, h))
               for s in (0, 1) for h in (False, True)):
            continue
        eligible.append(row)
        # A broad extra pool permits token-length-safe counterfactuals.
        if len(eligible) >= count * 3:
            break
    selected = []
    for i, row in enumerate(eligible):
        for offset in range(1, min(80, len(eligible))):
            donor = eligible[(i + offset) % len(eligible)]["answer"]
            if answer_text(donor) == answer_text(row["answer"]):
                continue
            edited = row["support"][:row["answer_start"]] + donor + row["support"][row["answer_end"]:]
            edited_ids = tokenizer.encode(edited, add_special_tokens=False)
            if len(edited_ids) <= 128 and answer_text(donor) not in answer_text(row["anchor"]):
                selected.append(dict(row, edited_support=edited, edited_support_ids=edited_ids,
                                     edited_answer=donor))
                break
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"Only {len(selected)} usable {corpus.split} episodes; requested {count}")
    return selected


def prepare(data, tokenizer_path, output):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "vera-corpus-episodes-v1", "seed": 20261013,
              "task": "Eight-word anchored three-word natural passage completion",
              "selection": "Deterministic, no model-outcome filtering; exclude cross-split overlapping articles",
              "data_sha256": sha256(Path(data) / "manifest.json"), "splits": {}}
    for split, count in (("train", 16384), ("validation", 512), ("test", 1000)):
        rows = select(Corpus(data, split), tokenizer, count, 20261013)
        path = output / f"{split}.jsonl"
        with path.open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=True) + "\n")
        report["splits"][split] = {"episodes": len(rows), "articles": len(set(r["article"] for r in rows)),
                                   "sha256": sha256(path)}
        print(json.dumps({"split": split, **report["splits"][split]}), flush=True)
    atomic_json(output / "manifest.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prepare(args.data, args.tokenizer, args.output)
