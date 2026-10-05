"""CPU-only tests for the persistent exact-search reference store."""

from pathlib import Path
import tempfile
import unittest

import torch

from vera_mem.vector_store import PersistentVectorDB


class VectorStoreTests(unittest.TestCase):
    def populated(self, top_k=2):
        store = PersistentVectorDB(3, 2, temperature=0.5, top_k=top_k)
        store.write("first", torch.tensor([2.0, 0.0, 0.0]), torch.tensor([1.0, 2.0]), 1)
        store.write("second", torch.tensor([0.0, 3.0, 0.0]), torch.tensor([8.0, 4.0]), 2)
        store.write("third", torch.tensor([0.0, 0.0, 1.0]), torch.tensor([-1.0, 3.0]), 3)
        return store

    def test_per_token_sparse_cosine_search(self):
        store = self.populated()
        queries = torch.tensor([[[3.0, 1.0, 0.0], [0.0, 1.0, 4.0]],
                                [[1.0, 2.0, 0.0], [0.0, 4.0, 1.0]]], requires_grad=True)
        before = store.hash()
        result = store.search(queries)
        self.assertEqual(result["mixed_value"].shape, (2, 2, 2))
        self.assertEqual(result["indices"].shape, (2, 2, 2))
        self.assertEqual([row[0] for row in result["ids"]], ["first", "third", "second", "second"])
        self.assertEqual(result["mixed_value"].device.type, "cpu")
        self.assertEqual(result["mixed_value"].dtype, torch.float32)
        self.assertFalse(result["mixed_value"].requires_grad)
        self.assertTrue(torch.allclose(result["weights"].sum(-1), torch.ones(2, 2)))
        self.assertTrue(torch.allclose(result["scores"][0, 0], torch.tensor([3.0, 1.0]) / (10 ** 0.5)))
        selected = store.values[result["indices"]]
        self.assertTrue(torch.allclose(result["mixed_value"], (selected * result["weights"].unsqueeze(-1)).sum(-2)))
        self.assertEqual(before, store.hash())
        result["mixed_value"].zero_()
        result["scores"].fill_(-999)
        result["ids"][0][0] = "tampered"
        self.assertEqual(before, store.hash())

    def test_append_update_stale_rejection_and_equal_timestamp(self):
        store = self.populated(top_k=1)
        store.write("second", torch.tensor([0.0, 3.0, 0.0]), torch.tensor([11.0, 12.0]), 5)
        self.assertEqual(len(store), 3)
        self.assertEqual(store.ids, ("first", "second", "third"))
        self.assertEqual(store.timestamps, (1, 5, 3))
        self.assertEqual(store.search(torch.tensor([0.0, 1.0, 0.0]))["mixed_value"].tolist(), [11.0, 12.0])
        before = store.hash()
        with self.assertRaisesRegex(ValueError, "Stale"):
            store.write("second", torch.tensor([1.0, 0.0, 0.0]), torch.ones(2), 4)
        self.assertEqual(before, store.hash())
        store.write("second", torch.tensor([0.0, 1.0, 0.0]), torch.tensor([13.0, 14.0]), 5)
        self.assertEqual(store.values[1].tolist(), [13.0, 14.0])

    def test_empty_store_and_empty_query_batch(self):
        store = PersistentVectorDB(3, 5)
        result = store.search(torch.zeros(2, 4, 3))
        self.assertEqual(result["mixed_value"].shape, (2, 4, 5))
        self.assertEqual(result["indices"].shape, (2, 4, 0))
        self.assertEqual(result["weights"].shape, (2, 4, 0))
        self.assertEqual(result["ids"], [[] for _ in range(8)])
        self.assertEqual(result["mixed_value"].sum().item(), 0)
        single = store.search(torch.zeros(3))
        self.assertEqual(single["mixed_value"].shape, (5,))
        self.assertEqual(single["indices"].shape, (0,))
        populated = self.populated()
        self.assertEqual(populated.search(torch.empty(0, 3))["indices"].shape, (0, 2))

    def test_snapshot_properties_and_input_copy_isolation(self):
        store = PersistentVectorDB(3, 2)
        key, value = torch.tensor([2.0, 1.0, 0.0]), torch.tensor([1.0, 7.0])
        store.write("x", key, value, 1.5)
        before = store.hash()
        key.zero_()
        value.zero_()
        store.keys.zero_()
        store.values.zero_()
        state = store.snapshot()
        state["keys"].zero_()
        state["values"].zero_()
        state["ids"][0] = "changed"
        state["timestamps"][0] = -1
        state["config"]["top_k"] = 999
        self.assertEqual(before, store.hash())
        self.assertEqual(store.resident_bytes(), {"key_bytes": 12, "value_bytes": 8, "total_bytes": 20})
        self.assertEqual(store.bytes(), store.resident_bytes())

    def test_atomic_save_weights_only_load_and_independence(self):
        store = self.populated()
        before = store.hash()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested/memory.pt"
            store.save(path)
            raw = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(raw["config"]["temperature"], 0.5)
            restored = PersistentVectorDB.load(path)
            self.assertEqual(before, restored.hash())
            self.assertEqual(before, store.hash())
            self.assertTrue(torch.equal(store.keys, restored.keys))
            self.assertTrue(torch.equal(store.values, restored.values))
            self.assertEqual(restored.timestamps, store.timestamps)
            raw["keys"].zero_()
            restored.write("new", torch.ones(3), torch.ones(2), 4)
            self.assertEqual(before, store.hash())
            self.assertEqual(len(restored), 4)
            restored.save(path)
            self.assertEqual(len(PersistentVectorDB.load(path)), 4)
            self.assertEqual(list(path.parent.glob("*.partial")), [])
            empty_path = Path(directory) / "empty.pt"
            PersistentVectorDB(3, 2).save(empty_path)
            self.assertEqual(len(PersistentVectorDB.load(empty_path)), 0)

    def test_timestamp_validation_and_failed_write_is_atomic(self):
        store = self.populated()
        before = store.hash()
        for invalid in (float("nan"), float("inf"), True, "tomorrow", None):
            with self.assertRaisesRegex(ValueError, "timestamp"):
                store.write("bad", torch.ones(3), torch.ones(2), invalid)
        for key, value in ((torch.zeros(3), torch.ones(2)),
                           (torch.ones(4), torch.ones(2)),
                           (torch.ones(3), torch.tensor([float("nan"), 1.0]))):
            with self.assertRaises(ValueError):
                store.write("first", key, value, 5)
        self.assertEqual(before, store.hash())
        with self.assertRaises(ValueError):
            store.search(torch.zeros(3))
        with self.assertRaises(ValueError):
            store.search(torch.tensor([float("inf"), 0.0, 1.0]))
        with self.assertRaises(ValueError):
            store.search(torch.ones(2))

    def test_malformed_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.pt"
            state = self.populated().snapshot()
            state["timestamps"][0] = float("nan")
            torch.save(state, path)
            with self.assertRaises(ValueError):
                PersistentVectorDB.load(path)
            state = self.populated().snapshot()
            state["keys"][0].zero_()
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, "unit-normalized"):
                PersistentVectorDB.load(path)

    def test_constructor_validation(self):
        for kwargs in ({"key_dim": 0}, {"value_dim": -2}, {"top_k": True},
                       {"temperature": 0}, {"temperature": float("nan")}):
            config = {"key_dim": 3, "value_dim": 2}
            config.update(kwargs)
            with self.assertRaises(ValueError):
                PersistentVectorDB(**config)


if __name__ == "__main__":
    unittest.main()
