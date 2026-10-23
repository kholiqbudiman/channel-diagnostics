"""Re-run ALL Paper 2 Table 3 baselines with proper sklearn macro_f1 logging.

The original scripts only logged accuracy. The F1 values in Table 3 are suspect
(likely weighted F1 mislabeled as macro F1). This script re-runs every config
with proper sklearn.metrics.f1_score(average='macro').

Table 3 configs (WaRP-C K=5):
  1. DWSep CE (64px)        — OrigDWSep, 10 seeds
  2. DWSep + CORD tau=2     — OrigDWSep Born-Again 2 gen, 10 seeds
  3. GhostNet CE (96px)     — GhostNet, 10 seeds
  4. InvRes CE (96px)       — InvResNet, 10 seeds
  5. InvRes BA3 (96px)      — InvResNet Born-Again 3 gen, 10 seeds
  6. ACRE-MSF CE (96px)     — ACRE_MSF, 10 seeds
  7. MNv3-Small (96px)      — MobileNetV3Tiny, 5 seeds
  8. MCUNet-style (96px)    — MobileNetV2Tiny, 5 seeds

Usage:
  cd ~/KD/Disertasi/Paper\ 3/hks-ot-artifact
  PYTHONPATH=src python3 ~/KD/wasterecycle/table3_rerun.py --gpus 0,1 --workers-per-gpu 5
"""
import argparse
import json
import logging
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset

from sklearn.metrics import f1_score, confusion_matrix


class ImageFolderNested(Dataset):
    """Like torchvision ImageFolder but traverses nested dirs (category/fine-class)."""
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s [%(process)d] %(message)s",
)
log = logging.getLogger(__name__)

# ─── Constants ──────────────────────────────────────────────────────────────
DATA_ROOT = None  # set via --data-root or defaults per-server
OUT_DIR = Path("/home/kholiq/KD/wasterecycle/results_table3_rerun")

K = 5
MAP_5 = {
    "bottle": list(range(0, 17)) + [23, 24, 25],
    "canister": [17],
    "cans": [18],
    "cardboard": [26, 27],
    "detergent": [19, 20, 21, 22],
}
CLASS_NAMES = ["bottle", "canister", "cans", "cardboard", "detergent"]

EPOCHS = 100
BATCH_SIZE = 128
PATIENCE = 20


# ─── Helpers ────────────────────────────────────────────────────────────────

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


# ═══════════════════════════════════════════════════════════════════════════
# ARCHITECTURES (copied verbatim from source files)
# ═══════════════════════════════════════════════════════════════════════════

# --- From arch_sweep_85.py ---

class DWSepBlock(nn.Module):
    def __init__(self, inc, outc, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(inc, inc, 3, stride=stride, padding=1, groups=inc, bias=False)
        self.bn1 = nn.BatchNorm2d(inc)
        self.pw = nn.Conv2d(inc, outc, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(outc)

    def forward(self, x):
        return F.relu6(self.bn2(self.pw(F.relu6(self.bn1(self.dw(x)), inplace=True))), inplace=True)


class OrigDWSep(nn.Module):
    """Original Paper 3 DWSep (14K baseline)"""
    def __init__(self, K=5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 14, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(14), nn.ReLU6(inplace=True))
        self.blocks = nn.Sequential(
            DWSepBlock(14, 20, 2), DWSepBlock(20, 28, 1), DWSepBlock(28, 40, 2),
            DWSepBlock(40, 56, 1), DWSepBlock(56, 80, 2))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(80, K)

    def forward(self, x):
        return self.fc(self.pool(self.blocks(self.stem(x))).flatten(1))


class InvResBlock_Sweep(nn.Module):
    """MobileNetV2-style inverted residual (from arch_sweep_85.py, expand=4 default)"""
    def __init__(self, inc, outc, stride=1, expand=4):
        super().__init__()
        mid = inc * expand
        self.conv = nn.Sequential(
            nn.Conv2d(inc, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU6(True),
            nn.Conv2d(mid, mid, 3, stride=stride, padding=1, groups=mid, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU6(True),
            nn.Conv2d(mid, outc, 1, bias=False), nn.BatchNorm2d(outc))
        self.skip = (stride == 1 and inc == outc)

    def forward(self, x):
        out = self.conv(x)
        return out + x if self.skip else out


class InvResNet(nn.Module):
    """Inverted residual network ~14K params (from arch_sweep_85.py)"""
    def __init__(self, K=5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU6(True))
        self.blocks = nn.Sequential(
            InvResBlock_Sweep(16, 16, stride=1, expand=2),
            InvResBlock_Sweep(16, 24, stride=2, expand=2),
            InvResBlock_Sweep(24, 24, stride=1, expand=2),
            InvResBlock_Sweep(24, 40, stride=2, expand=2),
            InvResBlock_Sweep(40, 64, stride=2, expand=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, K)

    def forward(self, x):
        return self.fc(self.pool(self.blocks(self.stem(x))).flatten(1))


class GhostModule(nn.Module):
    """GhostNet: generate cheap features via linear transform"""
    def __init__(self, inc, outc, ratio=2, kernel=1, dw_kernel=3):
        super().__init__()
        init_ch = outc // ratio
        cheap_ch = outc - init_ch
        self.primary = nn.Sequential(
            nn.Conv2d(inc, init_ch, kernel, bias=False), nn.BatchNorm2d(init_ch), nn.ReLU6(True))
        self.cheap = nn.Sequential(
            nn.Conv2d(init_ch, cheap_ch, dw_kernel, padding=dw_kernel // 2, groups=init_ch, bias=False),
            nn.BatchNorm2d(cheap_ch), nn.ReLU6(True))

    def forward(self, x):
        p = self.primary(x)
        return torch.cat([p, self.cheap(p)], dim=1)


class GhostNet(nn.Module):
    """GhostNet-Micro ~16K params (from arch_sweep_85.py)"""
    def __init__(self, K=5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU6(True))
        self.features = nn.Sequential(
            GhostModule(16, 32, ratio=2),
            nn.Conv2d(32, 32, 3, stride=2, padding=1, groups=32, bias=False), nn.BatchNorm2d(32), nn.ReLU6(True),
            GhostModule(32, 48, ratio=2),
            nn.Conv2d(48, 48, 3, stride=2, padding=1, groups=48, bias=False), nn.BatchNorm2d(48), nn.ReLU6(True),
            GhostModule(48, 64, ratio=2),
            nn.Conv2d(64, 64, 3, stride=2, padding=1, groups=64, bias=False), nn.BatchNorm2d(64), nn.ReLU6(True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, K)

    def forward(self, x):
        return self.fc(self.pool(self.features(self.stem(x))).flatten(1))


# --- From acre_baselines.py ---

class InvResBlock_Baselines(nn.Module):
    """InvRes block from acre_baselines.py (expand=3 default)"""
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


class MobileNetV2Tiny(nn.Module):
    """MobileNetV2 scaled to ~25K params (from acre_baselines.py)."""
    def __init__(self, num_classes=5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU6(True))
        self.blocks = nn.Sequential(
            InvResBlock_Baselines(16, 16, stride=1, expand=1),
            InvResBlock_Baselines(16, 24, stride=2, expand=3),
            InvResBlock_Baselines(24, 24, stride=1, expand=3),
            InvResBlock_Baselines(24, 32, stride=2, expand=3),
            InvResBlock_Baselines(32, 48, stride=2, expand=3),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(48, num_classes)

    def forward(self, x):
        return self.classifier(self.pool(self.blocks(self.stem(x))).flatten(1))


class SEBlock(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, ch // r), nn.ReLU(True),
            nn.Linear(ch // r, ch), nn.Hardsigmoid())

    def forward(self, x):
        return x * self.fc(x).unsqueeze(-1).unsqueeze(-1)


class MBv3Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expand=3, use_se=True):
        super().__init__()
        mid = inp * expand
        self.use_skip = (stride == 1 and inp == oup)
        layers = []
        if expand != 1:
            layers += [nn.Conv2d(inp, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.Hardswish(True)]
        layers += [nn.Conv2d(mid, mid, 3, stride=stride, padding=1, groups=mid, bias=False),
                   nn.BatchNorm2d(mid), nn.Hardswish(True)]
        if use_se:
            layers.append(SEBlock(mid))
        layers += [nn.Conv2d(mid, oup, 1, bias=False), nn.BatchNorm2d(oup)]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x) + x if self.use_skip else self.conv(x)


class MobileNetV3Tiny(nn.Module):
    """MobileNetV3-style with SE, Hardswish, scaled to ~25K (from acre_baselines.py)."""
    def __init__(self, num_classes=5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.Hardswish(True))
        self.blocks = nn.Sequential(
            MBv3Block(16, 16, stride=1, expand=1, use_se=False),
            MBv3Block(16, 24, stride=2, expand=2, use_se=False),
            MBv3Block(24, 24, stride=1, expand=2, use_se=True),
            MBv3Block(24, 32, stride=2, expand=2, use_se=True),
            MBv3Block(32, 48, stride=2, expand=2, use_se=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(48, num_classes)

    def forward(self, x):
        return self.classifier(self.pool(self.blocks(self.stem(x))).flatten(1))


# --- From acre_msf.py ---

class InvResBlock_MSF(nn.Module):
    """InvRes block from acre_msf.py (expand=3 default)"""
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


class ACRE_MSF(nn.Module):
    """ACRE with Multi-Scale Fusion (from acre_msf.py).

    Fuses texture (early block GAP) + semantic (late block GAP)
    features at classifier. +120 params vs single-scale.
    """
    def __init__(self, num_classes=5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU6(True))
        self.block1 = InvResBlock_MSF(16, 16, stride=1, expand=2)
        self.block2 = InvResBlock_MSF(16, 24, stride=2, expand=3)
        self.block3 = InvResBlock_MSF(24, 24, stride=1, expand=3)
        self.block4 = InvResBlock_MSF(24, 40, stride=2, expand=3)
        self.block5 = InvResBlock_MSF(40, 64, stride=2, expand=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(24 + 64, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        feat_texture = self.pool(x).flatten(1)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        feat_semantic = self.pool(x).flatten(1)
        fused = torch.cat([feat_texture, feat_semantic], dim=1)
        return self.classifier(fused)


# ═══════════════════════════════════════════════════════════════════════════
# MODEL BUILDERS
# ═══════════════════════════════════════════════════════════════════════════

def build_model(arch_name):
    builders = {
        "orig_dwsep": lambda: OrigDWSep(K=K),
        "ghostnet": lambda: GhostNet(K=K),
        "invresnet": lambda: InvResNet(K=K),
        "acre_msf": lambda: ACRE_MSF(num_classes=K),
        "mnv3_tiny": lambda: MobileNetV3Tiny(num_classes=K),
        "mbv2_tiny": lambda: MobileNetV2Tiny(num_classes=K),
    }
    return builders[arch_name]()


# ═══════════════════════════════════════════════════════════════════════════
# TABLE 3 CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════════

CONFIGS = {
    # 1. DWSep CE (64px) - 10 seeds
    "dwsep_ce_64": {
        "arch": "orig_dwsep", "resolution": 64, "ba_generations": 1,
        "n_seeds": 10, "description": "DWSep CE (64px)",
    },
    # 2. DWSep + CORD tau=2 - 10 seeds
    "dwsep_cord_t2_64": {
        "arch": "orig_dwsep", "resolution": 64, "ba_generations": 2,
        "n_seeds": 10, "description": "DWSep + CORD tau=2",
    },
    # 3. GhostNet CE (96px) - 10 seeds
    "ghostnet_ce_96": {
        "arch": "ghostnet", "resolution": 96, "ba_generations": 1,
        "n_seeds": 10, "description": "GhostNet CE (96px)",
    },
    # 4. InvRes CE (96px) - 10 seeds
    "invres_ce_96": {
        "arch": "invresnet", "resolution": 96, "ba_generations": 1,
        "n_seeds": 10, "description": "InvRes CE (96px)",
    },
    # 5. InvRes BA3 (96px) - 10 seeds
    "invres_ba3_96": {
        "arch": "invresnet", "resolution": 96, "ba_generations": 3,
        "n_seeds": 10, "description": "InvRes BA3 (96px)",
    },
    # 6. ACRE-MSF CE (96px) - 10 seeds
    "acre_msf_ce_96": {
        "arch": "acre_msf", "resolution": 96, "ba_generations": 1,
        "n_seeds": 10, "description": "ACRE-MSF CE (96px)",
    },
    # 7. MNv3-Small (96px) - 5 seeds
    "mnv3_small_96": {
        "arch": "mnv3_tiny", "resolution": 96, "ba_generations": 1,
        "n_seeds": 5, "description": "MNv3-Small (96px)",
    },
    # 8. MCUNet-style (96px) - 5 seeds
    "mcunet_style_96": {
        "arch": "mbv2_tiny", "resolution": 96, "ba_generations": 1,
        "n_seeds": 5, "description": "MCUNet-style (96px)",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def get_transforms(res):
    """Light augmentation (same as original training)."""
    tr_tf = transforms.Compose([
        transforms.Resize((res, res)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    te_tf = transforms.Compose([
        transforms.Resize((res, res)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return tr_tf, te_tf


def build_dataloaders(res):
    """Build train/val/test dataloaders."""
    tr_tf, te_tf = get_transforms(res)
    warpc_root = DATA_ROOT / "warpc" / "Warp-C"

    train_ds = ImageFolderNested(str(warpc_root / "train_crops"), transform=tr_tf)
    test_ds = ImageFolderNested(str(warpc_root / "test_crops"), transform=te_tf)

    # Split train into train/val (90/10)
    n = len(train_ds)
    n_val = int(n * 0.1)
    n_train = n - n_val
    gen = torch.Generator().manual_seed(42)
    train_sub, val_sub = torch.utils.data.random_split(train_ds, [n_train, n_val], generator=gen)

    # Val subset with test transforms (no augmentation)
    val_ds_actual = torch.utils.data.Subset(
        ImageFolderNested(str(warpc_root / "train_crops"), transform=te_tf),
        val_sub.indices,
    )

    trl = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True,
                     num_workers=2, pin_memory=True, drop_last=True)
    val = DataLoader(val_ds_actual, batch_size=256, shuffle=False, num_workers=2)
    tel = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)

    return trl, val, tel


# ═══════════════════════════════════════════════════════════════════════════
# EVALUATION (the whole point of this rerun)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_full(model, loader, dev, lmap):
    """Compute accuracy, macro_f1, weighted_f1, per_class_f1, confusion_matrix."""
    model.eval()
    lmap = lmap.to(dev)
    all_preds = []
    all_labels = []

    for batch in loader:
        x, y28 = batch[0].to(dev), batch[1].to(dev)
        yK = lmap[y28]
        preds = model(x).argmax(1)
        all_preds.append(preds.cpu())
        all_labels.append(yK.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    acc = 100.0 * (all_preds == all_labels).sum() / len(all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average="macro") * 100.0
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted") * 100.0
    per_class = f1_score(all_labels, all_preds, average=None) * 100.0
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(K)))

    per_class_dict = {CLASS_NAMES[i]: round(float(per_class[i]), 4) for i in range(K)}

    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_class_f1": per_class_dict,
        "confusion_matrix": cm.tolist(),
    }


@torch.no_grad()
def evaluate_val_acc(model, loader, dev, lmap):
    """Quick accuracy for early stopping."""
    model.eval()
    c = t = 0
    for batch in loader:
        x, y28 = batch[0].to(dev), batch[1].to(dev)
        yK = lmap[y28]
        c += (model(x).argmax(1) == yK).sum().item()
        t += yK.size(0)
    return 100.0 * c / t


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train_one_generation(model, trl, val_loader, dev, lmap, rid, epochs=EPOCHS,
                         teacher=None, T=2.0):
    """Train a single generation.

    If teacher is None: pure CE with label_smoothing=0.1
    If teacher is given: 0.5 * KL(student/T, teacher/T) * T^2 + 0.5 * CE(student, label, ls=0.1)
    """
    opt = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    sch = LambdaLR(opt, lambda e: cosine_lr(e, 5, epochs))

    lmap_d = lmap.to(dev)
    best_val = 0.0
    best_state = None
    patience_cnt = 0

    if teacher is not None:
        teacher.eval()
        teacher.to(dev)

    for ep in range(epochs):
        model.train()
        for batch in trl:
            x, y28 = batch[0].to(dev), batch[1].to(dev)
            yK = lmap_d[y28]
            opt.zero_grad()

            logits_s = model(x)

            if teacher is not None:
                with torch.no_grad():
                    logits_t = teacher(x)
                # KD loss
                kl_loss = F.kl_div(
                    F.log_softmax(logits_s / T, dim=1),
                    F.softmax(logits_t / T, dim=1),
                    reduction="batchmean",
                ) * (T * T)
                ce_loss = F.cross_entropy(logits_s, yK, label_smoothing=0.1)
                loss = 0.5 * kl_loss + 0.5 * ce_loss
            else:
                loss = F.cross_entropy(logits_s, yK, label_smoothing=0.1)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        sch.step()

        # Validation accuracy for early stopping
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

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val


def seed_all(seed):
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_single_job(config_name, cfg, seed, dev, lmap):
    """Run one (config, seed) job. Returns result dict."""
    run_id = f"{config_name}_s{seed}"
    t0 = time.time()

    seed_all(seed)

    res = cfg["resolution"]
    arch_name = cfg["arch"]
    ba_gens = cfg["ba_generations"]

    # Build dataloaders
    trl, val_loader, tel = build_dataloaders(res)
    lmap_d = lmap.to(dev)

    # Build model and count params
    model = build_model(arch_name).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"[{run_id}] arch={arch_name} params={n_params} res={res} BA_gens={ba_gens}")

    if ba_gens == 1:
        # Pure CE training
        model, best_val = train_one_generation(model, trl, val_loader, dev, lmap, run_id)
    else:
        # Born-Again / CORD: multiple generations
        # Gen 0: pure CE
        log.info(f"[{run_id}] Generation 0 (CE)")
        model, best_val = train_one_generation(model, trl, val_loader, dev, lmap,
                                               f"{run_id}_gen0")

        for gen_i in range(1, ba_gens):
            log.info(f"[{run_id}] Generation {gen_i} (KD from gen {gen_i - 1})")
            teacher = model
            teacher.eval()
            # New student (fresh init)
            seed_all(seed + gen_i * 1000)  # Different init per generation
            student = build_model(arch_name).to(dev)
            student, best_val = train_one_generation(
                student, trl, val_loader, dev, lmap,
                f"{run_id}_gen{gen_i}",
                teacher=teacher, T=2.0,
            )
            model = student

    # Final evaluation on test set
    metrics = evaluate_full(model, tel, dev, lmap)

    elapsed = time.time() - t0
    result = {
        "run_id": run_id,
        "config": config_name,
        "arch": arch_name,
        "params": n_params,
        "seed": seed,
        "resolution": res,
        "ba_generations": ba_gens,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "per_class_f1": metrics["per_class_f1"],
        "confusion_matrix": metrics["confusion_matrix"],
        "elapsed_s": round(elapsed, 1),
    }

    log.info(
        f"[{run_id}] DONE: acc={metrics['accuracy']:.2f}% "
        f"macro_f1={metrics['macro_f1']:.2f}% "
        f"weighted_f1={metrics['weighted_f1']:.2f}% "
        f"elapsed={elapsed:.0f}s"
    )

    return result


# ═══════════════════════════════════════════════════════════════════════════
# PARALLEL EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

def build_job_list():
    """Build flat list of (config_name, cfg, seed) jobs."""
    jobs = []
    for config_name, cfg in CONFIGS.items():
        n_seeds = cfg["n_seeds"]
        for seed in range(1, n_seeds + 1):
            jobs.append((config_name, cfg, seed))
    return jobs


def worker_fn(worker_id, gpu_id, assigned_jobs, lmap, data_root_str=None, out_dir_str=None):
    """Worker process: runs assigned jobs on specified GPU."""
    global DATA_ROOT, OUT_DIR
    if data_root_str:
        DATA_ROOT = Path(data_root_str)
    if out_dir_str:
        OUT_DIR = Path(out_dir_str)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    dev = torch.device("cuda:0")
    lmap = lmap.clone()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    worker_file = OUT_DIR / f"worker_{worker_id:02d}.json"

    # Load existing results for resume
    done = set()
    results = []
    if worker_file.exists():
        with open(worker_file) as f:
            results = json.load(f)
        done = {r["run_id"] for r in results}

    for config_name, cfg, seed in assigned_jobs:
        run_id = f"{config_name}_s{seed}"
        if run_id in done:
            log.info(f"[Worker {worker_id}] Skipping {run_id} (already done)")
            continue

        log.info(f"[Worker {worker_id}] Starting {run_id} on GPU {gpu_id}")
        try:
            result = run_single_job(config_name, cfg, seed, dev, lmap)
            results.append(result)
            done.add(run_id)
            # Save after each job
            with open(worker_file, "w") as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            log.error(f"[Worker {worker_id}] FAILED {run_id}: {e}")
            import traceback
            traceback.print_exc()

    log.info(f"[Worker {worker_id}] Finished all {len(assigned_jobs)} jobs")


def merge_results():
    """Merge all worker_XX.json files into results.json."""
    all_results = []
    seen = set()
    for wf in sorted(OUT_DIR.glob("worker_*.json")):
        with open(wf) as f:
            for r in json.load(f):
                if r["run_id"] not in seen:
                    all_results.append(r)
                    seen.add(r["run_id"])

    merged_file = OUT_DIR / "results.json"
    with open(merged_file, "w") as f:
        json.dump(all_results, f, indent=2)

    log.info(f"Merged {len(all_results)} results into {merged_file}")
    return all_results


def print_summary(results):
    """Print mean +/- std for each config."""
    by_config = defaultdict(list)
    for r in results:
        by_config[r["config"]].append(r)

    print("\n" + "=" * 100)
    print("  TABLE 3 RERUN SUMMARY")
    print("=" * 100)
    print(f"  {'Config':<25s} {'Arch':<12s} {'Params':>7s} {'Res':>4s} "
          f"{'N':>3s} {'Acc':>12s} {'Macro-F1':>12s} {'Weighted-F1':>12s}")
    print("-" * 100)

    for config_name in CONFIGS:
        runs = by_config.get(config_name, [])
        if not runs:
            print(f"  {config_name:<25s}  (no results)")
            continue

        cfg = CONFIGS[config_name]
        accs = np.array([r["accuracy"] for r in runs])
        mf1s = np.array([r["macro_f1"] for r in runs])
        wf1s = np.array([r["weighted_f1"] for r in runs])

        n = len(runs)
        if n > 1:
            acc_str = f"{accs.mean():.2f} +/- {accs.std(ddof=1):.2f}"
            mf1_str = f"{mf1s.mean():.2f} +/- {mf1s.std(ddof=1):.2f}"
            wf1_str = f"{wf1s.mean():.2f} +/- {wf1s.std(ddof=1):.2f}"
        else:
            acc_str = f"{accs[0]:.2f}"
            mf1_str = f"{mf1s[0]:.2f}"
            wf1_str = f"{wf1s[0]:.2f}"

        print(f"  {cfg['description']:<25s} {cfg['arch']:<12s} "
              f"{runs[0]['params']:>7d} {cfg['resolution']:>4d} "
              f"{n:>3d} {acc_str:>12s} {mf1_str:>12s} {wf1_str:>12s}")

    print("=" * 100)

    # Per-class F1 breakdown
    print("\n  PER-CLASS MACRO F1 BREAKDOWN (mean across seeds):")
    print(f"  {'Config':<25s} " + " ".join(f"{c:>10s}" for c in CLASS_NAMES))
    print("-" * 85)

    for config_name in CONFIGS:
        runs = by_config.get(config_name, [])
        if not runs:
            continue
        cfg = CONFIGS[config_name]
        per_class_means = {}
        for cls in CLASS_NAMES:
            vals = [r["per_class_f1"][cls] for r in runs]
            per_class_means[cls] = np.mean(vals)
        print(f"  {cfg['description']:<25s} " +
              " ".join(f"{per_class_means[c]:>10.2f}" for c in CLASS_NAMES))

    print("=" * 85)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Re-run Paper 2 Table 3 with proper macro F1")
    parser.add_argument("--gpus", type=str, default="0",
                        help="Comma-separated GPU IDs (e.g., '0,1')")
    parser.add_argument("--workers-per-gpu", type=int, default=5,
                        help="Number of parallel workers per GPU")
    parser.add_argument("--configs", type=str, default="all",
                        help="Comma-separated config names, or 'all'")
    parser.add_argument("--data-root", type=str,
                        default="/home/kholiq/KD/Disertasi/Paper 3/hks-ot-artifact/data/raw",
                        help="Path to data root containing warpc/Warp-C/")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Override output directory")
    args = parser.parse_args()

    global DATA_ROOT, OUT_DIR
    DATA_ROOT = Path(args.data_root)
    if args.out_dir:
        OUT_DIR = Path(args.out_dir)

    gpu_ids = [int(g) for g in args.gpus.split(",")]
    workers_per_gpu = args.workers_per_gpu
    total_workers = len(gpu_ids) * workers_per_gpu

    # Filter configs if specified
    if args.configs != "all":
        selected = [c.strip() for c in args.configs.split(",")]
        for c in selected:
            if c not in CONFIGS:
                raise ValueError(f"Unknown config: {c}. Available: {list(CONFIGS.keys())}")

    # Build job list
    all_jobs = build_job_list()
    if args.configs != "all":
        selected_set = set(c.strip() for c in args.configs.split(","))
        all_jobs = [(cn, cfg, s) for cn, cfg, s in all_jobs if cn in selected_set]

    log.info(f"Total jobs: {len(all_jobs)}")
    log.info(f"GPUs: {gpu_ids}, workers/gpu: {workers_per_gpu}, total_workers: {total_workers}")

    # Print param counts for all architectures
    log.info("Architecture param counts:")
    for arch_name in sorted(set(cfg["arch"] for cfg in CONFIGS.values())):
        m = build_model(arch_name)
        n = sum(p.numel() for p in m.parameters())
        log.info(f"  {arch_name}: {n:,d} params")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Distribute jobs round-robin across workers
    worker_jobs = [[] for _ in range(total_workers)]
    for i, job in enumerate(all_jobs):
        worker_jobs[i % total_workers].append(job)

    # Map worker_id -> gpu_id
    worker_gpu = {}
    for w in range(total_workers):
        gpu_idx = w // workers_per_gpu
        worker_gpu[w] = gpu_ids[gpu_idx]

    lmap = build_lmap()

    dr = str(DATA_ROOT)
    od = str(OUT_DIR)

    if total_workers == 1:
        # Single worker: run in main process
        worker_fn(0, gpu_ids[0], worker_jobs[0], lmap, dr, od)
    else:
        # Spawn workers
        mp.set_start_method("spawn", force=True)
        processes = []
        for w in range(total_workers):
            if not worker_jobs[w]:
                continue
            p = mp.Process(
                target=worker_fn,
                args=(w, worker_gpu[w], worker_jobs[w], lmap, dr, od),
            )
            p.start()
            processes.append(p)
            log.info(f"Spawned worker {w} on GPU {worker_gpu[w]} with {len(worker_jobs[w])} jobs")

        for p in processes:
            p.join()

    # Merge and summarize
    results = merge_results()
    print_summary(results)


if __name__ == "__main__":
    main()
