import argparse
import os
import sys
from typing import Tuple


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import flipsweep


def analyse_sweep(
    path: str,
    acc_cat_thresh: float,
    label_change_cat_thresh: float,
) -> Tuple[int, int, int]:

    res = flipsweep.load_sweep_result(path)

    total_bytes = len(res.retcoll_map)
    total_bits = total_bytes * 8

    flippable_bits = 0
    catastrophic_bits = 0

    for _byte_off, coll_list in res.retcoll_map.items():
        for coll in coll_list:
            # Each FlipResultColl corresponds to one bit position
            flippable_bits += 1

            # For our use we generally have a single binary/dataset per sweep.
            # Take the first entry; if there are more, treat catastrophic if
            # any of them meets the threshold.
            accs = list(coll.correct_pcts)
            label_changes = list(coll.top_label_change_pcts)

            is_cat = False
            for acc, lchg in zip(accs, label_changes):
                if acc <= acc_cat_thresh or lchg >= label_change_cat_thresh:
                    is_cat = True
                    break

            if is_cat:
                catastrophic_bits += 1

    return total_bits, flippable_bits, catastrophic_bits


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Compute catastrophic-bit statistics from BitShield sweep pickles. "
            "Intended for comparing unprotected vs protected binaries."
        )
    )
    parser.add_argument(
        "sweeps",
        nargs="+",
        help="Path(s) to *-sweep.pkl files produced by flipsweep.py",
    )
    parser.add_argument(
        "--acc-cat",
        type=float,
        default=30.0,
        help=(
            "Accuracy threshold for catastrophic flips: a flip is catastrophic "
            "if post-flip accuracy (in %) is <= this value (default: 30.0)."
        ),
    )
    parser.add_argument(
        "--label-change-cat",
        type=float,
        default=50.0,
        help=(
            "Label-change threshold for catastrophic flips: a flip is "
            "catastrophic if the percentage of changed top-1 labels is >= "
            "this value (default: 50.0)."
        ),
    )

    args = parser.parse_args()

    print(
        f"Using catastrophic thresholds: acc <= {args.acc_cat:.2f}%, "
        f"label_change >= {args.label_change_cat:.2f}%"
    )
    print()

    for sweep_path in args.sweeps:
        if not os.path.exists(sweep_path):
            print(f"[WARN] Sweep file not found: {sweep_path}")
            continue

        total_bits, flippable_bits, catastrophic_bits = analyse_sweep(
            sweep_path, args.acc_cat, args.label_change_cat
        )

        defended_bits = total_bits - flippable_bits

        sweep_name = os.path.basename(sweep_path)
        print(f"=== {sweep_name} ===")
        print(f"Total bits in swept region     : {total_bits}")
        print(f"Flippable bits (any recorded)  : {flippable_bits}")
        print(f"Defended bits (never recorded) : {defended_bits}")
        print(f"Catastrophic bits              : {catastrophic_bits}")

        if total_bits > 0:
            flippable_frac = flippable_bits / total_bits
            defended_frac = defended_bits / total_bits
            catastrophic_frac = catastrophic_bits / total_bits
        else:
            flippable_frac = defended_frac = catastrophic_frac = 0.0

        if flippable_bits > 0:
            catastrophic_given_flippable = catastrophic_bits / flippable_bits
        else:
            catastrophic_given_flippable = 0.0

        print(f"Flippable fraction             : {flippable_frac:.6f}")
        print(f"Defended fraction              : {defended_frac:.6f}")
        print(f"Catastrophic fraction          : {catastrophic_frac:.6f}")
        print(
            f"Catastrophic | flippable       : "
            f"{catastrophic_given_flippable:.6f}"
        )
        print()


if __name__ == "__main__":  # pragma: no cover
    main()
