"""Read-only gate and exploratory surrogate validation AFTER all locked runs.

Do not call on partial results to select models. Missing/failed runs stop analysis.
Does not queue training, change locks, write to the pilot, or read test images.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

from confusion_response import Reference, ConfusionResponse, heldout_indices, macro_f1


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def design_identifiability(protocol):
    models = [m for m in protocol["models"] if m["kind"] == "allocation"
              and m["residual"] and m["name"] != "base"]
    x = np.array([np.log(np.array(m["channels"])/[16, 24, 40, 64]) for m in models])
    singular = np.linalg.svd(x, compute_uv=False)
    return {"nonreference_allocations": len(models), "four_width_transfer_rank": int(np.linalg.matrix_rank(x)),
            "four_width_singular_values": singular.tolist(),
            "condition_number": float(singular[0]/singular[-1]),
            "five_block_transfer_rank": int(np.linalg.matrix_rank(x[:, [0, 1, 1, 2, 3]])),
            "warning": "Blocks 2 and 3 share the same width. Their independent allocation effects are not identifiable from this pilot. Ridge coefficients are not causal effects. Grow/shrink directions are incompletely covered."}


def reference_from_result(result):
    classes = result["classes"]
    k = len(classes)
    sensitivity, separation = np.zeros((5, k)), np.zeros((5, k, k))
    for s in range(5):
        stage = f"block{s+1}"
        masks = [x for x in result["channel_sensitivity"] if x["stage"] == stage]
        if sorted(x["mask_seed"] for x in masks) != [5101, 5102, 5103]:
            raise ValueError("Missing/duplicated locked masks")
        sensitivity[s] = np.mean([x["class_loss_delta"] for x in masks], axis=0)
        for p in result["representation"][stage]["pairwise"]:
            i, j = classes.index(p["class_a"]), classes.index(p["class_b"])
            separation[s, i, j] = separation[s, j, i] = p["distance_over_within_variance"]
    return Reference(result["job"]["model"]["channels"],
                     result["probe_metrics"]["confusion_matrix"],
                     result["probe_metrics"]["class_loss"], sensitivity, separation)


def load_complete(pilot, runs):
    lock_path = pilot / "protocol_lock.json"
    lock = read(lock_path)
    changed = [name for name, sha in lock["sources"].items() if digest(pilot/name) != sha]
    if changed:
        raise ValueError(f"Locked inputs changed: {changed}")
    protocol = read(pilot / "protocol.json")
    jobs = protocol["jobs"]
    if len({j["id"] for j in jobs}) != len(jobs):
        raise ValueError("Duplicate jobs")
    missing = [j["id"] for j in jobs if not (runs/j["id"]/"result.json").is_file()]
    if missing:
        return None, {"state": "WAITING_FOR_COMPLETE_COHORT", "completed": len(jobs)-len(missing),
                      "total": len(jobs), "missing": missing,
                      "protocol_lock_sha256": digest(lock_path),
                      "scientific_results_computed": False}
    # Only after the complete-cohort gate do we read performance outcomes.
    costs = read(pilot/"preflight.json")["model_costs"]
    manifests = {name: read(pilot/f"data_{name}.json") for name in protocol["datasets"]}
    results, sources = {}, []
    for job in jobs:
        folder = runs/job["id"]
        result_path = folder/"result.json"
        result = read(result_path)
        manifest = manifests[job["dataset"]]
        if result["job"] != job or result["classes"] != manifest["classes"]:
            raise ValueError(f"Job/class identity mismatch: {job['id']}")
        if result["cost"] != costs[job["dataset"]][job["model"]["name"]]:
            raise ValueError("Budget accounting mismatch")
        for filename, hashkey in (("best.pt", "checkpoint_sha256"),
                                  ("initial_state.pt", "initialization_file_sha256")):
            if digest(folder/filename) != result[hashkey]:
                raise ValueError(f"Checkpoint provenance mismatch: {job['id']}")
        if digest(pilot/f"data_{job['dataset']}.json") != result["split_manifest_sha256"]:
            raise ValueError("Split fingerprint mismatch")
        features_path = folder/"probe_features.npz"
        with np.load(features_path, allow_pickle=False) as data:
            y, pred = data["labels"], data["logits"].argmax(1)
            expected_y = np.array([manifest["samples"][i][1] for i in manifest["split"]["probe"]])
            if not np.array_equal(y, expected_y):
                raise ValueError("Probe sample order differs from locked manifest")
            cm = np.zeros((len(result["classes"]),)*2, dtype=int)
            np.add.at(cm, (y, pred), 1)
            if not np.array_equal(cm, result["probe_metrics"]["confusion_matrix"]):
                raise ValueError("Stored predictions/confusion disagree")
            if abs(macro_f1(cm)*100-result["probe_metrics"]["macro_f1_pct"]) > 1e-8:
                raise ValueError("Stored metric/confusion disagree")
        results[job["id"]] = result
        sources.append({"job_id": job["id"], "result_sha256": digest(result_path),
                        "probe_features_sha256": digest(features_path),
                        "checkpoint_sha256": result["checkpoint_sha256"]})
    # Import exact locked cost formula. No torch/model instantiation.
    spec = importlib.util.spec_from_file_location("locked_budget", pilot/"budget_design.py")
    budget = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(budget)
    records = []
    for job in jobs:
        config = job["model"]
        if config["kind"] != "allocation" or not config["residual"] or config["name"] == "base":
            continue
        result = results[job["id"]]
        base = results[f"{job['dataset']}_base_s{job['seed']}"]
        ref = reference_from_result(base)
        expected_cost = budget.allocation_cost(config["channels"], classes=len(ref.cm))
        if any(expected_cost[k] != result["cost"][k] for k in ("params", "macs")):
            raise ValueError("Formula and runtime accounting disagree")
        records.append({"id": job["id"], "dataset": job["dataset"], "seed": job["seed"],
                        "channels": config["channels"], "reference": ref,
                        "target_cm": np.array(result["probe_metrics"]["confusion_matrix"]),
                        "cost": result["cost"]})
    return records, {"state": "COMPLETE_DEVELOPMENT_COHORT", "completed": len(jobs), "total": len(jobs),
                     "protocol_lock_sha256": digest(lock_path), "sources": sources,
                     "design_identifiability": design_identifiability(protocol),
                     "sum_recorded_run_elapsed_s": sum(x["elapsed_s"] for x in results.values()),
                     "scientific_results_computed": False}


def validate_surrogate(records):
    output = []
    for axis in ("allocation", "dataset"):
        for key, train, test in heldout_indices(records, axis):
            for mode in ("architecture_only", "frequency_only", "class_agnostic", "class_pair"):
                model = ConfusionResponse(mode=mode, ridge=1.).fit([records[i] for i in train])
                predictions = []
                for i in test:
                    r = records[i]
                    pred = model.predict_cm(r["reference"], r["channels"])
                    predictions.append({"id": r["id"], "dataset": r["dataset"], "seed": r["seed"],
                                        "predicted_macro_f1_pct": macro_f1(pred)*100,
                                        "observed_macro_f1_pct": macro_f1(r["target_cm"])*100,
                                        "reference_macro_f1_pct": macro_f1(r["reference"].cm)*100})
                selection = []
                contexts = {(p["dataset"], p["seed"]) for p in predictions}
                for dataset, seed in sorted(contexts):
                    subset = [p for p in predictions if (p["dataset"], p["seed"]) == (dataset, seed)]
                    if len(subset) < 2:
                        continue  # allocation holdout has one candidate; no rank claim
                    base_score = subset[0]["reference_macro_f1_pct"]
                    choices = subset + [{"id": "manual_base", "predicted_macro_f1_pct": base_score,
                                         "observed_macro_f1_pct": base_score}]
                    chosen = sorted(choices, key=lambda p: (-p["predicted_macro_f1_pct"], p["id"]))[0]
                    oracle = max(p["observed_macro_f1_pct"] for p in choices)
                    selection.append({"dataset": dataset, "seed": seed, "selected": chosen["id"],
                                      "gain_vs_manual_base_pp": chosen["observed_macro_f1_pct"]-base_score,
                                      "development_oracle_regret_pp": oracle-chosen["observed_macro_f1_pct"],
                                      "expected_uniform_random_score_pct": float(np.mean([p["observed_macro_f1_pct"] for p in choices]))})
                output.append({"heldout_axis": axis, "heldout_group": key, "mode": mode,
                               "training_ids": model.training_ids, "predictions": predictions,
                               "mae_pp": float(np.mean([abs(p["predicted_macro_f1_pct"]-p["observed_macro_f1_pct"])
                                                         for p in predictions])),
                               "selection": selection})
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Protect the pilot even if a caller passes a mistaken output path.
    if args.output.resolve().is_relative_to(args.pilot.resolve()):
        parser.error("Output must be outside the locked pilot directory")
    if args.output.exists():
        parser.error("Output already exists; preserve previous reports")
    records, report = load_complete(args.pilot, args.runs)
    report["analysis_source_sha256"] = {p.name: digest(p) for p in
                                        (Path(__file__), Path(__file__).with_name("confusion_response.py"))}
    report["scope"] = "Exploratory method development only. No independent confirmation, novelty proof, or calibrated uncertainty."
    if records is not None:
        report["surrogate_validation"] = validate_surrogate(records)
        report["scientific_results_computed"] = True
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(report["state"], f"{report['completed']}/{report['total']}")


if __name__ == "__main__":
    main()
