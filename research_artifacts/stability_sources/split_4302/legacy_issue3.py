"""Issue #3: DWSep at 96px baseline — isolate architecture vs resolution effect.

The paper's +14.67pp headline (ACRE-MSF 96px vs DWSep 64px) confounds
resolution (64→96) with architecture changes. This experiment creates
a DWSep-only model at 96px with ~20K params to disentangle:

  Resolution effect:   DWSep-64px (14K) vs DWSep-96px (20K)
  Architecture effect:  DWSep-96px (20K) vs ACRE-Base-96px (20K)
  Full effect:          DWSep-64px (14K) vs ACRE-MSF-96px (25K)

DWSep-96px channels: stem=16, blocks=[24,40,56,72,120], fc=120→5
Param count: ~20,237 (matches ACRE-Base's 20,245)

Mode B: val-based checkpoint (90/10 split), same as table3_rerun.py

Usage:
  cd ~/KD/Disertasi/Paper\ 3/hks-ot-artifact
  PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python3 ~/KD/wasterecycle/issue3_dwsep96.py
"""
import argparse, json, logging, math, random, time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from sklearn.metrics import f1_score, confusion_matrix as sk_cm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)

DATA_ROOT = Path("/home/kholiq/KD/Disertasi/Paper 3/hks-ot-artifact/data/raw")
OUT_DIR = Path("/home/kholiq/KD/wasterecycle/results_dwsep96")
K = 5
MAP_5 = {
    "bottle": list(range(0, 17)) + [23, 24, 25],
    "canister": [17], "cans": [18],
    "cardboard": [26, 27], "detergent": [19, 20, 21, 22],
}
CLASS_NAMES = ["bottle", "canister", "cans", "cardboard", "detergent"]
EPOCHS = 100
BATCH_SIZE = 128
PATIENCE = 20


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_lmap():
    lmap = torch.zeros(28, dtype=torch.long)
    for i, (_, idx) in enumerate(MAP_5.items()):
        for j in idx:
            lmap[j] = i
    return lmap


def cosine_lr(ep, warmup, total):
    if ep < warmup:
        return max(1e-3, (ep + 1) / max(1, warmup))
    return 0.5 * (1 + math.cos(math.pi * (ep - warmup) / max(1, total - warmup)))


# ═══════════════════════════════════════════════════════════════════════
# IMAGEFOLDERNESTED (self-contained, no hksot dependency)
# ═══════════════════════════════════════════════════════════════════════

class ImageFolderNested(Dataset):
    IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(self, root, transform=None, fine_level=2):
        self.root = Path(root)
        self.transform = transform
        leaf_dirs = sorted([p for p in self.root.glob("*/" * fine_level) if p.is_dir()])
        self.classes = [str(p.relative_to(self.root)) for p in leaf_dirs]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = []
        for cls in self.classes:
            cls_idx = self.class_to_idx[cls]
            for img in sorted((self.root / cls).rglob("*")):
                if img.suffix.lower() in self.IMG_EXT:
                    self.samples.append((img, cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ═══════════════════════════════════════════════════════════════════════
# ARCHITECTURES
# ═══════════════════════════════════════════════════════════════════════

class DWSepBlock(nn.Module):
    def __init__(self, inc, outc, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(inc, inc, 3, stride=stride, padding=1, groups=inc, bias=False)
        self.bn1 = nn.BatchNorm2d(inc)
        self.pw = nn.Conv2d(inc, outc, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(outc)

    def forward(self, x):
        return F.relu6(self.bn2(self.pw(F.relu6(self.bn1(self.dw(x)), inplace=True))), inplace=True)


class DWSep96(nn.Module):
    """DWSep at 96px, ~20K params. Widened channels to match ACRE-Base param count."""
    def __init__(self, K=5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU6(inplace=True))
        self.blocks = nn.Sequential(
            DWSepBlock(16, 24, 2),
            DWSepBlock(24, 40, 1),
            DWSepBlock(40, 56, 2),
            DWSepBlock(56, 72, 1),
            DWSepBlock(72, 120, 2))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(120, K)

    def forward(self, x):
        return self.fc(self.pool(self.blocks(self.stem(x))).flatten(1))


class InvResBlock(nn.Module):
    def __init__(self, inp, oup, stride=1, expand=3):
        super().__init__()
        mid = inp * expand
        self.use_skip = (stride == 1 and inp == oup)
        layers = []
        if expand != 1:
            layers += [nn.Conv2d(inp, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU6(True)]
        layers += [nn.Conv2d(mid, mid, 3, stride=stride, padding=1, groups=mid, bias=False),
                   nn.BatchNorm2d(mid), nn.ReLU6(True)]
        layers += [nn.Conv2d(mid, oup, 1, bias=False), nn.BatchNorm2d(oup)]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x) + x if self.use_skip else self.conv(x)


class ACRE_Base(nn.Module):
    """ACRE-Base: single-scale GAP, uniform 2x expansion. ~20,245 params."""
    def __init__(self, num_classes=5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU6(True))
        self.block1 = InvResBlock(16, 16, stride=1, expand=2)
        self.block2 = InvResBlock(16, 24, stride=2, expand=2)
        self.block3 = InvResBlock(24, 24, stride=1, expand=2)
        self.block4 = InvResBlock(24, 40, stride=2, expand=2)
        self.block5 = InvResBlock(40, 64, stride=2, expand=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return self.classifier(self.pool(x).flatten(1))


ARCHS = {
    "dwsep_96": lambda: DWSep96(K),
    "acre_base": lambda: ACRE_Base(K),
}


# ═══════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════

def build_dataloaders(res, data_root):
    tr_tf = transforms.Compose([
        transforms.Resize((res, res)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    te_tf = transforms.Compose([
        transforms.Resize((res, res)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    warpc_root = data_root / "warpc" / "Warp-C"
    train_ds = ImageFolderNested(str(warpc_root / "train_crops"), transform=tr_tf)
    test_ds = ImageFolderNested(str(warpc_root / "test_crops"), transform=te_tf)

    n = len(train_ds)
    n_val = int(n * 0.1)
    n_train = n - n_val
    gen = torch.Generator().manual_seed(42)
    train_sub, val_sub = torch.utils.data.random_split(train_ds, [n_train, n_val], generator=gen)

    val_ds = torch.utils.data.Subset(
        ImageFolderNested(str(warpc_root / "train_crops"), transform=te_tf),
        val_sub.indices)

    trl = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True,
                     num_workers=2, pin_memory=True, drop_last=True)
    val = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2)
    tel = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)
    return trl, val, tel


# ═══════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_full(model, loader, dev, lmap):
    model.eval()
    lmap = lmap.to(dev)
    all_preds, all_labels = [], []
    for batch in loader:
        x, y28 = batch[0].to(dev), batch[1].to(dev)
        yK = lmap[y28]
        all_preds.append(model(x).argmax(1).cpu())
        all_labels.append(yK.cpu())

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    acc = 100.0 * (preds == labels).sum() / len(labels)
    macro_f1 = f1_score(labels, preds, average="macro") * 100.0
    weighted_f1 = f1_score(labels, preds, average="weighted") * 100.0
    per_class = f1_score(labels, preds, average=None) * 100.0
    cm = sk_cm(labels, preds, labels=list(range(K)))

    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_class_f1": {CLASS_NAMES[i]: round(float(per_class[i]), 4) for i in range(K)},
        "confusion_matrix": cm.tolist(),
    }


@torch.no_grad()
def evaluate_val_acc(model, loader, dev, lmap):
    model.eval()
    c = t = 0
    for batch in loader:
        x, y28 = batch[0].to(dev), batch[1].to(dev)
        yK = lmap[y28]
        c += (model(x).argmax(1) == yK).sum().item()
        t += yK.size(0)
    return 100.0 * c / t


# ═══════════════════════════════════════════════════════════════════════
# TRAINING — MODE B (val-based checkpoint)
# ═══════════════════════════════════════════════════════════════════════

def train_model(model, trl, val_loader, dev, lmap, rid, epochs=EPOCHS):
    opt = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    sch = LambdaLR(opt, lambda e: cosine_lr(e, 5, epochs))
    lmap_d = lmap.to(dev)
    best_val = 0.0
    best_state = None
    patience_cnt = 0

    for ep in range(epochs):
        model.train()
        for batch in trl:
            x, y28 = batch[0].to(dev), batch[1].to(dev)
            yK = lmap_d[y28]
            opt.zero_grad()
            loss = F.cross_entropy(model(x), yK, label_smoothing=0.1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()

        va = evaluate_val_acc(model, val_loader, dev, lmap_d)
        if va > best_val:
            best_val = va
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1

        if ep % 20 == 0 or ep == epochs - 1:
            log.info(f"[{rid}] ep={ep} val={va:.2f}% best_val={best_val:.2f}%")

        if patience_cnt >= PATIENCE:
            log.info(f"[{rid}] Early stopping at ep={ep}")
            break

    if best_state:
        model.load_state_dict(best_state)
    return best_val


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--archs", default="dwsep_96,acre_base")
    parser.add_argument("--split-id", default="", help="Suffix for output file to avoid race between parallel workers")
    parser.add_argument("--data-root", type=str, default=None)
    args = parser.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in args.seeds.split(",")]
    arch_names = [a.strip() for a in args.archs.split(",")]

    data_root = Path(args.data_root) if args.data_root else DATA_ROOT
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lmap = build_lmap()

    log.info("Architecture param counts:")
    for aname in arch_names:
        m = ARCHS[aname]()
        n = sum(p.numel() for p in m.parameters())
        log.info(f"  {aname}: {n:,d} params")

    trl, val_loader, tel = build_dataloaders(96, data_root)

    sfx = f"_{args.split_id}" if args.split_id else ""
    for aname in arch_names:
        rf = OUT_DIR / f"results_{aname}{sfx}.json"
        done, results = set(), []
        if rf.exists():
            results = json.load(open(rf))
            done = {r["run_id"] for r in results}

        for seed in seeds:
            rid = f"{aname}_s{seed}"
            if rid in done:
                continue

            t0 = time.time()
            seed_all(seed)

            model = ARCHS[aname]().to(dev)
            np_ = sum(p.numel() for p in model.parameters())
            log.info(f">>> [{rid}] {aname} params={np_:,}")

            best_val = train_model(model, trl, val_loader, dev, lmap, rid)
            metrics = evaluate_full(model, tel, dev, lmap)
            elapsed = time.time() - t0

            log.info(f"[{rid}] DONE: acc={metrics['accuracy']:.2f}% "
                     f"macro_f1={metrics['macro_f1']:.2f}% "
                     f"weighted_f1={metrics['weighted_f1']:.2f}% "
                     f"elapsed={elapsed:.0f}s")

            r = {"run_id": rid, "arch": aname, "params": np_,
                 "resolution": 96, "seed": seed,
                 "accuracy": metrics["accuracy"],
                 "macro_f1": metrics["macro_f1"],
                 "weighted_f1": metrics["weighted_f1"],
                 "per_class_f1": metrics["per_class_f1"],
                 "confusion_matrix": metrics["confusion_matrix"],
                 "elapsed_s": round(elapsed, 1)}
            results.append(r)
            done.add(rid)
            with open(rf, "w") as f:
                json.dump(results, f, indent=2)

    # Summary — merge all per-arch result files
    all_results = []
    for p in sorted(OUT_DIR.glob("results_*.json")):
        all_results.extend(json.load(open(p)))

    log.info("\n" + "=" * 90)
    log.info("ARCHITECTURE EFFECT ISOLATION: DWSep-96px vs ACRE-Base-96px")
    log.info("=" * 90)

    by_arch = defaultdict(list)
    for r in all_results:
        by_arch[r["arch"]].append(r)

    log.info(f"{'Arch':<15s} {'Params':>7s} {'Res':>4s} {'N':>3s} "
             f"{'Acc':>12s} {'Macro-F1':>12s} {'Weighted-F1':>12s}")
    log.info("-" * 80)

    for aname in arch_names:
        if aname not in by_arch:
            continue
        rs = by_arch[aname]
        accs = np.array([r["accuracy"] for r in rs])
        mf1s = np.array([r["macro_f1"] for r in rs])
        wf1s = np.array([r["weighted_f1"] for r in rs])
        n = len(rs)
        p = rs[0]["params"]
        if n > 1:
            log.info(f"  {aname:<13s} {p:>7,} {96:>4d} {n:>3d} "
                     f"{accs.mean():.2f}+/-{accs.std(ddof=1):.2f}  "
                     f"{mf1s.mean():.2f}+/-{mf1s.std(ddof=1):.2f}  "
                     f"{wf1s.mean():.2f}+/-{wf1s.std(ddof=1):.2f}")
        else:
            log.info(f"  {aname:<13s} {p:>7,} {96:>4d} {n:>3d} {accs[0]:.2f}%")

    # Per-class F1 breakdown
    log.info(f"\nPer-class F1 (mean):")
    log.info(f"{'Arch':<15s} " + " ".join(f"{c:>10s}" for c in CLASS_NAMES))
    for aname in arch_names:
        if aname not in by_arch:
            continue
        rs = by_arch[aname]
        pcm = {c: np.mean([r["per_class_f1"][c] for r in rs]) for c in CLASS_NAMES}
        log.info(f"  {aname:<13s} " + " ".join(f"{pcm[c]:>10.2f}" for c in CLASS_NAMES))

    log.info("Done!")


if __name__ == "__main__":
    main()
