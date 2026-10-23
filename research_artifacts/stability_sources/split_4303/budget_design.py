"""Generate a label-free, finite allocation experiment; no training/search scores."""
import argparse
import itertools
import json
from pathlib import Path


def allocation_cost(channels, expansions=(2, 2, 2, 2, 2), classes=5, resolution=96):
    a, b, c, d = channels
    size = (resolution + 1) // 2
    params, macs = 29 * a, size * size * 27 * a
    inp = a
    layers = []
    for out, stride, expand in zip((a, b, b, c, d), (1, 2, 1, 2, 2), expansions):
        mid = inp * expand
        end = (size + stride - 1) // stride
        p = 9 * mid + mid * out + 2 * mid + 2 * out
        m = end * end * (9 * mid + mid * out)
        if expand != 1:
            p += inp * mid + 2 * mid
            m += size * size * inp * mid
        layers.append({"params": p, "macs": m})
        params += p
        macs += m
        inp, size = out, end
    return {"params": params + d * classes + classes,
            "macs": macs + d * classes, "block_costs": layers}


def build_design():
    base = allocation_cost((16, 24, 40, 64))
    feasible = []
    grid = itertools.product(range(8, 26, 2), range(12, 42, 2),
                             range(16, 68, 4), range(24, 100, 4))
    for channels in grid:
        if list(channels) != sorted(channels):
            continue
        cost = allocation_cost(channels)
        if abs(cost["params"] / base["params"] - 1) <= .01 and abs(cost["macs"] / base["macs"] - 1) <= .01:
            name = "base" if channels == (16, 24, 40, 64) else "allocation_" + "_".join(map(str, channels))
            feasible.append({"name": name, "kind": "allocation", "channels": list(channels),
                             "expansions": [2] * 5, "residual": True, "cost_k5": cost})
    feasible.sort(key=lambda item: (item["name"] != "base", item["channels"]))
    return {"design": "Every monotone allocation in the specified finite grid satisfying both +/-1% bands; selected without performance or class-error information.",
            "grid": {"stem": list(range(8, 26, 2)), "second": list(range(12, 42, 2)),
                     "third": list(range(16, 68, 4)), "last": list(range(24, 100, 4))},
            "tolerance_fraction": .01, "reference_k5": base,
            "models": feasible + [
                {"name": "base_no_residual", "kind": "allocation", "channels": [16, 24, 40, 64],
                 "expansions": [2] * 5, "residual": False, "cost_k5": base},
                {"name": "dwsep96", "kind": "dwsep96"},
                {"name": "mcunet_style", "kind": "mcunet_style"}],
            "counting": "Conv2d+Linear MACs only; all trainable parameters including affine BN/bias. Residual additions, normalization, activations and pooling excluded from MACs; FLOPs=2*MACs."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build_design(), indent=2) + "\n")
    print("Saved all", len(build_design()["models"]), "models")
