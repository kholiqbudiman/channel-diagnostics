"""Existing operators; allocation variants are experimental controls, not new operators."""
import torch
from torch import nn
from legacy_issue3 import ACRE_Base, DWSep96, InvResBlock
from legacy_table3 import MobileNetV2Tiny


class AllocationNet(ACRE_Base):
    def __init__(self, config, classes):
        nn.Module.__init__(self)
        a, b, c, d = config["channels"]
        self.stem = nn.Sequential(nn.Conv2d(3, a, 3, 2, 1, bias=False),
                                  nn.BatchNorm2d(a), nn.ReLU6(True))
        inp = a
        for i, (out, stride, expand) in enumerate(zip((a, b, b, c, d), (1, 2, 1, 2, 2), config["expansions"]), 1):
            block = InvResBlock(inp, out, stride, expand)
            block.use_skip = block.use_skip and config["residual"]
            setattr(self, f"block{i}", block)
            inp = out
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(d, classes)


def build_model(config, classes):
    if config["kind"] == "allocation":
        return AllocationNet(config, classes)
    if config["kind"] == "dwsep96":
        return DWSep96(classes)
    if config["kind"] == "mcunet_style":
        return MobileNetV2Tiny(classes)
    raise ValueError(config)


def stage_modules(model):
    if isinstance(model, AllocationNet):
        return [(f"block{i}", getattr(model, f"block{i}")) for i in range(1, 6)]
    return [(f"block{i+1}", block) for i, block in enumerate(model.blocks)]


def count_model(model, resolution=96):
    model.eval()
    macs = 0
    handles = []
    def hook(module, inputs, output):
        nonlocal macs
        if isinstance(module, nn.Conv2d):
            macs += output.numel() * (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
        elif isinstance(module, nn.Linear):
            macs += output.numel() * module.in_features
    try:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                handles.append(module.register_forward_hook(hook))
        with torch.no_grad():
            model(torch.zeros(1, 3, resolution, resolution))
    finally:
        for handle in handles:
            handle.remove()
    return {"params": sum(p.numel() for p in model.parameters()), "macs": macs, "flops_convention": 2 * macs}
