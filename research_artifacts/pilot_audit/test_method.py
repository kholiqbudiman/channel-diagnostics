"""Synthetic implementation checks. These are NOT ACRE experimental results."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from confusion_response import (Reference, ConfusionResponse, feasible,
                                heldout_indices, macro_f1, pair_features)
from analyze_complete_pilot import load_complete


def example_reference(k=3):
    cm = np.eye(k)*70 + np.ones((k, k))*10
    return Reference([16, 24, 40, 64], cm, np.linspace(.3, 1., k),
                     np.arange(5*k).reshape(5, k)/100.,
                     np.ones((5, k, k))*.5)


def example_records():
    records = []
    for task, k in (("task_a", 3), ("task_b", 5)):
        for channels in ([14, 30, 36, 52], [16, 24, 36, 84]):
            for seed in (1, 2):
                ref = example_reference(k)
                target = ref.cm.copy()
                target[0, 0] += 2
                target[0, 1] -= 2
                records.append({"id": f"{task}_{channels}_{seed}", "dataset": task,
                                "seed": seed, "channels": channels,
                                "reference": ref, "target_cm": target})
    return records


class MethodTests(unittest.TestCase):
    def test_macro_f1_depends_on_confusion_destinations(self):
        first = np.array([[80, 20, 0], [0, 8, 2], [0, 2, 8]])
        second = np.array([[80, 0, 20], [2, 8, 0], [2, 0, 8]])
        np.testing.assert_array_equal(first.sum(1), second.sum(1))
        np.testing.assert_array_equal(np.diag(first), np.diag(second))
        self.assertGreater(abs(macro_f1(first)-macro_f1(second)), .02)

    def test_rejects_missing_class_support(self):
        with self.assertRaises(ValueError):
            macro_f1([[5, 0], [0, 0]])

    def test_joint_budget_filter(self):
        reference_cost = {"params": 20245, "macs": 9055616}
        self.assertTrue(feasible(reference_cost, reference_cost))
        self.assertFalse(feasible({"params": 20237, "macs": 3339384}, reference_cost))
        self.assertFalse(feasible({"params": 22261, "macs": 9022704}, reference_cost))
        model = ConfusionResponse().fit(example_records())
        candidates = [{"name": "base", "channels": [16, 24, 40, 64], "cost": reference_cost},
                      {"name": "bad", "channels": [14, 30, 36, 52],
                       "cost": {"params": 30000, "macs": 9055616}}]
        self.assertEqual([r["name"] for r in model.rank(example_reference(), candidates, reference_cost)], ["base"])

    def test_identity_transfer_is_exact(self):
        ref = example_reference()
        for mode in ("architecture_only", "frequency_only", "class_agnostic", "class_pair"):
            self.assertTrue((pair_features(ref, ref.channels, mode) == 0).all())
            model = ConfusionResponse(mode=mode).fit(example_records())
            np.testing.assert_array_equal(model.predict_cm(ref, ref.channels), ref.cm)

    def test_confusion_predictions_preserve_supports_and_finiteness(self):
        ref = example_reference()
        model = ConfusionResponse().fit(example_records())
        model.coef *= 1e6  # adversarial numerical case
        pred = model.predict_cm(ref, [14, 30, 36, 52])
        self.assertTrue(np.isfinite(pred).all())
        self.assertTrue((pred >= 0).all())
        np.testing.assert_allclose(pred.sum(1), ref.cm.sum(1))

    def test_semantic_class_permutation_invariance(self):
        records = [r for r in example_records() if r["dataset"] == "task_a"]
        model = ConfusionResponse().fit(records)
        ref = records[0]["reference"]
        channels = [14, 30, 36, 52]
        permutation = [2, 0, 1]
        perm_ref = Reference(ref.channels, ref.cm[np.ix_(permutation, permutation)],
                             ref.loss[permutation], ref.sensitivity[:, permutation],
                             ref.separation[:, permutation][:, :, permutation])
        original = model.predict_cm(ref, channels)
        perm_pred = model.predict_cm(perm_ref, channels)
        np.testing.assert_allclose(perm_pred, original[np.ix_(permutation, permutation)])

    def test_holdout_excludes_whole_allocation_and_task(self):
        records = example_records()
        for axis in ("allocation", "dataset"):
            for _, train, test in heldout_indices(records, axis):
                key = lambda r: tuple(r["channels"]) if axis == "allocation" else r["dataset"]
                self.assertFalse({key(records[i]) for i in train} & {key(records[i]) for i in test})
                self.assertEqual(len(train)+len(test), len(records))

    def test_paired_support_mismatch_is_rejected(self):
        records = example_records()
        records[0]["target_cm"][0, 0] += 1
        with self.assertRaises(ValueError):
            ConfusionResponse().fit(records)

    def test_missing_cohort_does_not_read_partial_outcomes(self):
        with tempfile.TemporaryDirectory() as tmp:
            pilot = Path(tmp)/"pilot"
            runs = Path(tmp)/"runs"
            pilot.mkdir()
            (runs/"a").mkdir(parents=True)
            (pilot/"protocol_lock.json").write_text(json.dumps({"sources": {}}))
            (pilot/"protocol.json").write_text(json.dumps({"jobs": [{"id": "a"}, {"id": "b"}]}))
            (runs/"a"/"result.json").write_text("INVALID PARTIAL OUTCOME MUST NOT BE READ")
            records, state = load_complete(pilot, runs)
            self.assertIsNone(records)
            self.assertFalse(state["scientific_results_computed"])
            self.assertEqual(state["completed"], 1)

    def test_changed_lock_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            pilot = Path(tmp)
            (pilot/"source.py").write_text("changed")
            (pilot/"protocol_lock.json").write_text(json.dumps({"sources": {"source.py": "wrong"}}))
            with self.assertRaisesRegex(ValueError, "Locked inputs changed"):
                load_complete(pilot, pilot/"runs")


if __name__ == "__main__":
    unittest.main()
