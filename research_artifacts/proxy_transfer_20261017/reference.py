"""Development pilot: training/validation/diagnostics only. Never reads a test directory."""
import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from models import build_model, count_model, stage_modules

HERE = Path(__file__).resolve().parent
CLASSES = ["bottle", "canister", "cans", "cardboard", "detergent"]
EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def discover(root, dataset):
    if root.name in ("test", "test_crops", "val", "val_crops"):
        raise ValueError("Pilot cannot read test/official evaluation directories")
    if dataset == "warpc":
        leaves = sorted(p for p in root.glob("*/*") if p.is_dir())
        if len(leaves) != 28:
            raise ValueError(f"Expected 28 fine-class directories, found {len(leaves)}")
        for leaf in leaves:
            if leaf.parent.name not in CLASSES:
                raise ValueError(f"Unknown semantic parent class: {leaf}")
        classes = CLASSES
        folder_labels = [(leaf, CLASSES.index(leaf.parent.name)) for leaf in leaves]
    else:
        leaves = sorted(p for p in root.iterdir() if p.is_dir())
        classes = [p.name for p in leaves]
        if classes != ["cardboard", "glass", "metal", "paper", "plastic", "trash"]:
            raise ValueError(f"Unexpected TrashNet classes: {classes}")
        folder_labels = list(zip(leaves, range(len(leaves))))
    samples = [(str(p.relative_to(root)), label) for leaf, label in folder_labels
               for p in sorted(leaf.rglob("*")) if p.suffix.lower() in EXT]
    return samples, classes


def split_indices(samples, seed):
    rng = np.random.default_rng(seed)
    parts = {"fit": [], "selection": [], "probe": []}
    labels = np.array([s[1] for s in samples])
    for label in sorted(set(labels)):
        ids = np.flatnonzero(labels == label)
        rng.shuffle(ids)
        n = max(2, int(len(ids) * .1))
        if len(ids) <= 2 * n + 2:
            raise ValueError("Too few samples for three-way split")
        parts["selection"].extend(ids[:n].tolist())
        parts["probe"].extend(ids[n:2*n].tolist())
        parts["fit"].extend(ids[2*n:].tolist())
    return {key: sorted(value) for key, value in parts.items()}


class Images(Dataset):
    def __init__(self, root, samples, ids, augment=False):
        self.root, self.samples, self.ids = root, samples, ids
        ops = [transforms.Resize((96, 96))]
        if augment:
            ops += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(15),
                    transforms.ColorJitter(.2, .2, .2)]
        ops += [transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])]
        self.transform = transforms.Compose(ops)
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, i):
        path, label = self.samples[self.ids[i]]
        with Image.open(self.root / path) as image:
            return self.transform(image.convert("RGB")), label


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def make_loader(ds, shuffle, seed, workers=2):
    return DataLoader(ds, batch_size=64 if shuffle else 128, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed), worker_init_fn=seed_worker,
                      num_workers=workers, pin_memory=True, drop_last=False)


def metrics(logits, labels, k):
    cm = confusion_matrix(labels.numpy(), logits.argmax(1).numpy(), labels=list(range(k)))
    losses = nn.functional.cross_entropy(logits, labels, reduction="none")
    actual, predicted = cm.sum(1), cm.sum(0)
    tp = np.diag(cm)
    return {"macro_f1_pct": float(f1_score(labels.numpy(), logits.argmax(1).numpy(), average="macro", labels=list(range(k)), zero_division=0) * 100),
            "accuracy_pct": float(tp.sum() / cm.sum() * 100), "confusion_matrix": cm.tolist(),
            "class_recall_pct": np.divide(tp * 100, actual, out=np.zeros(k), where=actual>0).tolist(),
            "class_precision_pct": np.divide(tp * 100, predicted, out=np.zeros(k), where=predicted>0).tolist(),
            "class_loss": [float(losses[labels == c].mean()) for c in range(k)]}


@torch.no_grad()
def evaluate(model, loader, device, collect=False):
    model.eval()
    vectors, handles, logits, labels = {}, [], [], []
    if collect:
        for name, module in stage_modules(model):
            vectors[name] = []
            def hook(_, inputs, output, name=name):
                vectors[name].append(output.mean((2, 3)).detach().cpu())
            handles.append(module.register_forward_hook(hook))
    try:
        for x, y in loader:
            logits.append(model(x.to(device)).cpu())
            labels.append(y)
    finally:
        for handle in handles:
            handle.remove()
    return torch.cat(logits), torch.cat(labels), {key: torch.cat(value) for key, value in vectors.items()}


def representation_diagnostics(features, labels, classes):
    output = {}
    for stage, z in features.items():
        z = nn.functional.normalize(z.double(), dim=1)
        centers = torch.stack([z[labels == c].mean(0) for c in range(len(classes))])
        within = torch.stack([((z[labels == c] - centers[c])**2).sum(1).mean() for c in range(len(classes))])
        pairs = []
        for a in range(len(classes)):
            for b in range(a + 1, len(classes)):
                distance = (centers[a] - centers[b]).square().sum()
                pairs.append({"class_a": classes[a], "class_b": classes[b],
                              "centroid_distance_sq": float(distance),
                              "distance_over_within_variance": float(distance / (within[a] + within[b] + 1e-12))})
        output[stage] = {"feature_width": z.shape[1], "pairwise": pairs,
                         "within_class_variance": dict(zip(classes, within.tolist()))}
    return output


def verify_lock():
    lock = json.loads((HERE / "protocol_lock.json").read_text())
    for name, expected in lock["sources"].items():
        if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Locked source changed: {name}")
    return lock


def run(job, root, out):
    verify_lock()
    result = out / job["id"] / "result.json"
    if result.exists():
        raise ValueError("Existing completed job cannot be overwritten")
    folder = result.parent
    folder.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    manifest = json.loads((HERE / f"data_{job['dataset']}.json").read_text())
    samples, classes = discover(root, job["dataset"])
    if samples != [tuple(s) for s in manifest["samples"]]:
        raise ValueError("Dataset enumeration changed since preflight")
    # Check every image against the preflight content fingerprint before training.
    for relative, expected in manifest["image_sha256"].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Image changed: {relative}")
    split = manifest["split"]
    seed = job["seed"]
    torch.manual_seed(seed)
    model = build_model(job["model"], len(classes))
    cost = count_model(model)
    expected_cost = json.loads((HERE / "preflight.json").read_text())["model_costs"][job["dataset"]][job["model"]["name"]]
    if cost != expected_cost:
        raise ValueError("Model accounting differs from preflight")
    torch.save({"model": model.state_dict(), "job": job}, folder / "initial_state.pt")
    initialization_hash = hashlib.sha256((folder / "initial_state.pt").read_bytes()).hexdigest()
    # Architecture initialization consumes RNG differently. Reset the data stream afterwards.
    random.seed(seed + 100000)
    np.random.seed(seed + 100000)
    torch.manual_seed(seed + 100000)
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires remote CUDA; CPU-only mode is preflight only")
    device = torch.device("cuda:0")
    model.to(device)
    train = make_loader(Images(root, samples, split["fit"], True), True, seed + 100000)
    selection = make_loader(Images(root, samples, split["selection"]), False, 4301)
    probe = make_loader(Images(root, samples, split["probe"]), False, 4301)
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    schedule = torch.optim.lr_scheduler.LambdaLR(opt, lambda ep: (ep+1)/5 if ep<5 else .5*(1+math.cos(math.pi*(ep-5)/95)))
    best, best_epoch, bad = -math.inf, None, 0
    history, started = [], time.time()
    for epoch in range(100):
        model.train()
        for x, y in train:
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x.to(device)), y.to(device), label_smoothing=.1)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
        schedule.step()
        logits, labels, _ = evaluate(model, selection, device)
        score = metrics(logits, labels, len(classes))["macro_f1_pct"]
        history.append({"epoch": epoch + 1, "selection_macro_f1_pct": score})
        if score > best:
            best, best_epoch, bad = score, epoch + 1, 0
            torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "job": job, "classes": classes, "cost": cost, "epoch": best_epoch,
                        "selection_macro_f1_pct": best}, folder / "best.pt")
        else:
            bad += 1
        atomic_json(folder / "progress.json", {"epoch": epoch+1, "elapsed_s": time.time()-started, "state": "TRAINING"})
        if bad >= 20:
            break
    checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    baseline_logits, labels, features = evaluate(model, probe, device, collect=True)
    baseline = metrics(baseline_logits, labels, len(classes))
    np.savez_compressed(folder / "probe_features.npz", labels=labels.numpy(), logits=baseline_logits.numpy(),
                        **{name: z.numpy() for name, z in features.items()})
    sensitivities = []
    for stage, module in stage_modules(model):
        width = features[stage].shape[1]
        for mask_seed in (5101, 5102, 5103):
            removed = np.random.default_rng(mask_seed).choice(width, max(1, round(width*.1)), replace=False)
            def mask_hook(_, inputs, output):
                mask = torch.ones(output.shape[1], device=output.device)
                mask[torch.as_tensor(removed, device=output.device)] = 0
                return output * mask[None, :, None, None]
            handle = module.register_forward_hook(mask_hook)
            try:
                perturbed, yp, _ = evaluate(model, probe, device)
            finally:
                handle.remove()
            if not torch.equal(labels, yp):
                raise ValueError("Probe order changed")
            diagnostic = metrics(perturbed, yp, len(classes))
            sensitivities.append({"stage": stage, "removed_channels": removed.tolist(), "mask_seed": mask_seed,
                                  "removed_fraction": len(removed)/width,
                                  "macro_f1_delta_pp": diagnostic["macro_f1_pct"] - baseline["macro_f1_pct"],
                                  "class_loss_delta": (np.array(diagnostic["class_loss"]) - baseline["class_loss"]).tolist(),
                                  "class_recall_delta_pp": (np.array(diagnostic["class_recall_pct"]) - baseline["class_recall_pct"]).tolist()})
    atomic_json(result, {"job": job, "classes": classes, "cost": cost, "best_epoch": best_epoch,
                        "selection_macro_f1_pct": best, "probe_metrics": baseline,
                        "representation": representation_diagnostics(features, labels, classes),
                        "channel_sensitivity": sensitivities, "history": history,
                        "checkpoint_sha256": hashlib.sha256((folder/"best.pt").read_bytes()).hexdigest(),
                        "initialization_file_sha256": initialization_hash,
                        "split_manifest_sha256": hashlib.sha256((HERE/f"data_{job['dataset']}.json").read_bytes()).hexdigest(),
                        "elapsed_s": time.time()-started, "scope": "Development pilot; no test access. Probe diagnostics are observational. No confirmatory significance claim."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output-root", type=Path, default=HERE / "runs")
    args = parser.parse_args()
    protocol = json.loads((HERE / "protocol.json").read_text())
    job = next(j for j in protocol["jobs"] if j["id"] == args.job_id)
    run(job, Path(protocol["datasets"][job["dataset"]]), args.output_root)
