#!/usr/bin/env python3
"""Random train/val/test split without grouping (stratified by label for classification).

Writes split.json and split_assignment.csv (every structure is its own group)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", required=True,
                   help="label CSV: id column + 'label' (classification) or id column + target columns (regression)")
    p.add_argument("--task", choices=["classification", "regression"], default="classification")
    p.add_argument("--outdir", default="data/split_random", help="output folder (default: data/split_random)")
    p.add_argument("--id_col", default="id")
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--test_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    lab = pd.read_csv(args.labels)
    if args.id_col not in lab.columns:
        sys.exit(f"id column '{args.id_col}' not in {args.labels}: {list(lab.columns)}")
    df = pd.DataFrame({"id": lab[args.id_col].astype(str)})
    if args.task == "classification":
        if "label" not in lab.columns:
            sys.exit(f"classification needs a 'label' column; found {list(lab.columns)}")
        strat = lab["label"].to_numpy()
    else:
        strat = None
    n_val, n_test = round(len(df) * args.val_frac), round(len(df) * args.test_frac)
    idx = np.arange(len(df))
    rest, test = train_test_split(idx, test_size=n_test, stratify=strat, random_state=args.seed)
    train, val = train_test_split(rest, test_size=n_val, random_state=args.seed,
                                  stratify=None if strat is None else strat[rest])
    df["split"] = ""
    df.loc[train, "split"], df.loc[val, "split"], df.loc[test, "split"] = "train", "val", "test"
    df["group_id"] = np.arange(len(df))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    js = {s: sorted(df.loc[df["split"] == s, "id"]) for s in ("train", "val", "test")}
    meta = dict(seed=args.seed, task=args.task, grouping="none", n_structures=len(df),
                counts={k: len(v) for k, v in js.items()})
    (outdir / "split.json").write_text(json.dumps({**meta, **js}, indent=1), encoding="utf-8")
    df.to_csv(outdir / "split_assignment.csv", index=False)
    print({k: len(v) for k, v in js.items()})
    print(f"Saved split.json and split_assignment.csv to {outdir}")


if __name__ == "__main__":
    main()
