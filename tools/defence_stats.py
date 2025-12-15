#! /usr/bin/env python3

import flipsweep


def defence_stats(label: str, pkl_path: str) -> None:
    res = flipsweep.load_sweep_result(pkl_path)

    # Each key in retcoll_map is a byte offset; value is list of FlipResultColls
    total_bytes = len(res.retcoll_map)
    total_bits = total_bytes * 8

    # For each byte, we only have FlipResultColls for bits that PASSED
    # the thresholds (i.e., flips that were considered interesting enough
    # to store as attacks).
    flippable_bits = sum(len(coll_list) for coll_list in res.retcoll_map.values())
    defended_bits = total_bits - flippable_bits

    print(f"=== {label} ===")
    print(f"Total bytes with at least one considered flip: {total_bytes}")
    print(f"Total bits in those bytes: {total_bits}")
    print(
        f"Flippable bits (attacks that got through thresholds): "
        f"{flippable_bits} ({(flippable_bits / total_bits * 100) if total_bits else 0:.2f}%)"
    )
    print(
        f"Defended bits (flips filtered out by SIG/CIG thresholds): "
        f"{defended_bits} ({(defended_bits / total_bits * 100) if total_bits else 0:.2f}%)"
    )
    print()


def main() -> None:
    defence_stats(
        "QresnetClean",
        "results/sweep/tvm-main-QresnetClean-CIFAR10-ncnp-nd-sweep.pkl",
    )
    defence_stats(
        "QresnetBD",
        "results/sweep/tvm-main-QresnetBD-CIFAR10-ncnp-nd-sweep.pkl",
    )


if __name__ == "__main__":
    main()
