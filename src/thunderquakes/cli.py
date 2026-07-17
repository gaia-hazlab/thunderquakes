"""Command-line entry point.

    thunderquakes stations --region OK
    thunderquakes characterize
    thunderquakes train --config configs/seismic_only.yaml

Thin dispatcher; each subcommand delegates to the corresponding module. Most
subcommands are WS-stubbed until their workstream lands (see ROADMAP.md).
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="thunderquakes", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_st = sub.add_parser("stations", help="Build the seismic+infrasound inventory for a region")
    p_st.add_argument("--region", required=True, choices=["OK", "PNW", "AK"])

    sub.add_parser("characterize", help="WS1 per-class signal characterization")

    p_tr = sub.add_parser("train", help="WS4 train the CNN classifier")
    p_tr.add_argument("--config", required=True)

    args = parser.parse_args(argv)

    if args.command == "stations":
        from thunderquakes.stations import build_inventory

        inv = build_inventory(args.region)
        print(inv.to_string(index=False))
        return 0
    if args.command == "characterize":
        raise SystemExit("WS1 not implemented yet — see ROADMAP.md (Signal characterization).")
    if args.command == "train":
        raise SystemExit("WS4 not yet implemented — see ROADMAP.md / issue for 'CNN training'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
