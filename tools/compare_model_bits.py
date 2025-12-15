import argparse
import os
from typing import Dict, Tuple

import numpy as np
import onnx
from onnx import numpy_helper
import torch


def _load_torch_tensors(path: str) -> Dict[str, torch.Tensor]:

    obj = torch.load(path, map_location="cpu")

    if isinstance(obj, torch.nn.Module):
        state_dict = obj.state_dict()
    elif isinstance(obj, dict) and "state_dict" in obj and isinstance(
        obj["state_dict"], dict
    ):
        state_dict = obj["state_dict"]
    elif isinstance(obj, dict):
        state_dict = obj
    else:
        raise ValueError(f"Unsupported PyTorch checkpoint type in {path!r}: {type(obj)}")

    tensors: Dict[str, torch.Tensor] = {}
    for name, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            tensors[name] = value.detach().cpu()

    if not tensors:
        raise ValueError(f"No tensors found in PyTorch checkpoint {path!r}")

    return tensors


def _load_onnx_tensors(path: str) -> Dict[str, np.ndarray]:

    model = onnx.load(path)
    tensors: Dict[str, np.ndarray] = {}

    for init in model.graph.initializer:
        tensors[init.name] = numpy_helper.to_array(init)

    if not tensors:
        raise ValueError(f"No initializers found in ONNX model {path!r}")

    return tensors


def _count_bit_diffs(a: np.ndarray, b: np.ndarray) -> Tuple[int, int]:

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    a_bytes = np.ascontiguousarray(a).view(np.uint8)
    b_bytes = np.ascontiguousarray(b).view(np.uint8)

    if a_bytes.shape != b_bytes.shape:
        raise ValueError(f"Byte-shape mismatch: {a_bytes.shape} vs {b_bytes.shape}")

    diff = np.bitwise_xor(a_bytes, b_bytes)
    # Count set bits via unpackbits; fast enough for typical model sizes
    diff_bits = int(np.unpackbits(diff).sum())
    total_bits = int(a_bytes.size * 8)
    return diff_bits, total_bits


def _compare_collections(
    coll_a: Dict[str, np.ndarray], coll_b: Dict[str, np.ndarray]
) -> Tuple[int, int, list]:

    common = sorted(set(coll_a.keys()) & set(coll_b.keys()))
    if not common:
        raise ValueError("No common tensor names between the two models")

    total_diff = 0
    total_bits = 0
    per_tensor = []

    for name in common:
        a = coll_a[name]
        b = coll_b[name]

        if isinstance(a, torch.Tensor):
            a_np = a.cpu().numpy()
        else:
            a_np = np.asarray(a)

        if isinstance(b, torch.Tensor):
            b_np = b.cpu().numpy()
        else:
            b_np = np.asarray(b)

        if a_np.shape != b_np.shape:
            # Skip mismatched shapes; they are unlikely to be the flipped weight
            continue

        diff_bits, bits = _count_bit_diffs(a_np, b_np)
        total_diff += diff_bits
        total_bits += bits
        if diff_bits > 0:
            per_tensor.append((name, diff_bits, bits))

    per_tensor.sort(key=lambda x: x[1], reverse=True)
    return total_diff, total_bits, per_tensor


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Compare two models (PyTorch .pth/.pt or ONNX .onnx) at the bit level "
            "to see how many bits differ."
        )
    )
    parser.add_argument("model_a", help="Path to clean / reference model")
    parser.add_argument("model_b", help="Path to backdoored / modified model")
    parser.add_argument(
        "--format",
        choices=["auto", "torch", "onnx"],
        default="auto",
        help=(
            "Model format. 'auto' infers from file extension: "
            ".pth/.pt -> torch, .onnx -> onnx."
        ),
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Show the top-N tensors with the most differing bits (default: 10)",
    )

    args = parser.parse_args()

    fmt = args.format
    ext_a = os.path.splitext(args.model_a)[1].lower()
    ext_b = os.path.splitext(args.model_b)[1].lower()

    if fmt == "auto":
        if ext_a != ext_b:
            raise SystemExit(
                "When using auto format detection, both models must have the same "
                f"extension (got {ext_a!r} vs {ext_b!r})."
            )
        if ext_a in {".pth", ".pt"}:
            fmt = "torch"
        elif ext_a == ".onnx":
            fmt = "onnx"
        else:
            raise SystemExit(
                f"Cannot auto-detect format from extension {ext_a!r}; "
                "use --format explicitly."
            )

    if fmt == "torch":
        coll_a = _load_torch_tensors(args.model_a)
        coll_b = _load_torch_tensors(args.model_b)
    elif fmt == "onnx":
        coll_a = _load_onnx_tensors(args.model_a)
        coll_b = _load_onnx_tensors(args.model_b)
    else:
        raise SystemExit(f"Unsupported format: {fmt}")

    total_diff, total_bits, per_tensor = _compare_collections(coll_a, coll_b)

    if total_bits == 0:
        raise SystemExit("No comparable tensors found between the two models")

    print(f"Total bits compared: {total_bits}")
    print(f"Total differing bits: {total_diff}")
    print(f"Fraction differing: {total_diff / total_bits:.6e}")

    if total_diff == 0:
        print("No bit differences found.")
        return

    print()
    print(f"Top {min(args.top, len(per_tensor))} tensors by differing bits:")
    for name, diff_bits, bits in per_tensor[: args.top]:
        frac = diff_bits / bits if bits else 0.0
        print(f"  {name}: {diff_bits} / {bits} bits differ ({frac:.6e})")


if __name__ == "__main__":  # pragma: no cover
    main()
