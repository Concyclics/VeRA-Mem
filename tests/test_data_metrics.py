"""Run with PYTHONPATH=src python -m unittest discover -s tests."""

from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from vera_mem.data import Example, MEMORY_WORDS, prepare_medmcqa, synthetic_dataset
from vera_mem.metrics import exact_match, load_examples, normalize_answer, save_examples, token_f1


class SyntheticTests(unittest.TestCase):
    def test_reproducible_balanced_disjoint_and_not_in_query(self):
        data = synthetic_dataset(seed=31, n_train=32, n_control=16)
        self.assertEqual(data, synthetic_dataset(seed=31, n_train=32, n_control=16))
        self.assertNotEqual(data, synthetic_dataset(seed=32, n_train=32, n_control=16))
        all_examples = data["stream"] + data["control"]
        self.assertEqual(len({example.id for example in all_examples}), 48)
        self.assertEqual(set(Counter(e.answer for e in data["stream"]).values()), {2})
        self.assertEqual(set(e.answer for e in data["control"]), set(MEMORY_WORDS))
        for example in all_examples:
            self.assertIn(example.answer, example.support)
            self.assertNotIn(example.answer, example.question.split())
            self.assertNotIn(example.answer, example.paraphrase.split())
            self.assertNotEqual(example.question, example.paraphrase)
        self.assertTrue(all(e.metadata["never_written"] for e in data["control"]))

    def test_empty_and_invalid_counts(self):
        self.assertEqual(synthetic_dataset(n_train=0, n_control=0), {"stream": [], "control": []})
        for bad in (-1, 0.5, True):
            with self.assertRaises(ValueError):
                synthetic_dataset(n_train=bad)

    def test_metadata_not_shared(self):
        a = Example("a", "q", "a", "s", "p")
        b = Example("b", "q", "a", "s", "p")
        a.metadata["test"] = True
        self.assertEqual(b.metadata, {})


class MetricTests(unittest.TestCase):
    def test_normalized_em_is_not_substring(self):
        self.assertEqual(normalize_answer("  A.\n"), "a")
        self.assertEqual(exact_match("Apple!", "apple"), 1.0)
        self.assertEqual(exact_match("the answer is apple", "apple"), 0.0)
        self.assertEqual(exact_match("A", ""), 0.0)

    def test_multiset_f1_and_empty(self):
        self.assertAlmostEqual(token_f1("apple apple river", "apple river"), 0.8)
        self.assertEqual(token_f1("apple", "river"), 0.0)
        self.assertEqual(token_f1("", ""), 1.0)
        self.assertEqual(token_f1("", "apple"), 0.0)

    def test_json_roundtrip_and_duplicate_rejection(self):
        examples = synthetic_dataset(n_train=3)["stream"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested/examples.jsonl"
            save_examples(path, examples)
            self.assertEqual(examples, load_examples(path))
            save_examples(path, [examples[0], examples[0]])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_examples(path)


def row(identifier, question, cop=0, choice_type="single"):
    return {"id": identifier, "question": question, "opa": "red", "opb": "blue",
            "opc": "green", "opd": "yellow", "cop": cop,
            "choice_type": choice_type, "exp": "SECRET EXPLANATION"}


class MedmcqaTests(unittest.TestCase):
    def test_pinned_download_no_test_split_no_overlap_no_explanation(self):
        revision = "a" * 40
        source = {"sha": revision, "siblings": [
            {"rfilename": f"data/{split}-00000-of-00001.parquet"}
            for split in ("train", "validation", "test")]}
        downloads = []

        def fake_download(url, destination):
            downloads.append(url)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"fixture parquet")

        def fake_read(paths):
            if "validation" in paths[0].name:
                return [row("overlap", " TRAIN QUESTION "), row("v1", "New question", 3)]
            return [row("t1", "train question"), row("t2", "train question")]

        with tempfile.TemporaryDirectory() as directory, \
             patch("vera_mem.data._read_json_url", return_value=source) as api, \
             patch("vera_mem.data._download_file", side_effect=fake_download), \
             patch("vera_mem.data._read_parquet_rows", side_effect=fake_read):
            data = prepare_medmcqa(directory, n_train=1, n_control=1)
            self.assertEqual(data["stream"][0].answer, "A")
            self.assertEqual(data["control"][0].answer, "D")
            self.assertEqual(len(downloads), 2)
            self.assertTrue(all(f"/resolve/{revision}/" in url for url in downloads))
            self.assertTrue(all("test-" not in url for url in downloads))
            for example in data["stream"] + data["control"]:
                self.assertNotIn("SECRET EXPLANATION", example.support + example.question)
            manifest = json.loads((Path(directory) / "manifest.json").read_text())
            self.assertEqual(manifest["source_revision"], revision)
            self.assertEqual(manifest["label_schema"]["source_values"], [0, 1, 2, 3])
            self.assertEqual(len(manifest["source_files"]), 2)
            self.assertEqual(load_examples(Path(directory) / "stream.jsonl"), data["stream"])
            self.assertEqual(prepare_medmcqa(directory, n_train=1, n_control=1), data)
            self.assertIn(f"/revision/{revision}", api.call_args.args[0])
            self.assertEqual(len(downloads), 2)
            (Path(directory) / "source" / revision / source["siblings"][0]["rfilename"]).write_bytes(b"bad")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                prepare_medmcqa(directory, n_train=1, n_control=1)

    def test_invalid_label_is_not_silently_shifted(self):
        from vera_mem.data import _medmcqa_candidates
        with self.assertRaisesRegex(ValueError, "0..3"):
            _medmcqa_candidates([row("bad", "question", cop=4)])


if __name__ == "__main__":
    unittest.main()
