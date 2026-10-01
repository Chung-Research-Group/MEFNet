#!/usr/bin/env python3
"""Group-based train/val/test split: structures sharing a group never end up in different subsets.

Groups come from an existing group column, from MOFid (metal|linker|topology), or, if neither is given,
each structure is its own group (equivalent to a random split)."""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

METALS = {
    "Li", "Be", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe",
    "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc",
    "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Cs", "Ba", "La", "Ce", "Pr",
    "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf",
    "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Th", "U", "Si",
}
BAD = {"", "nan", "none", "null", "na", "n/a", "unknown", "error", "?", "-"}
_MOFID = re.compile(r"\sMOFid-v[12]\.(\S+)")
_BRACKET = re.compile(r"\[([^\]]+)\]")
_ELEMENT = re.compile(r"([A-Z][a-z]?)")



def make_strata(y, bins):
    """Class codes for categorical labels, quantile bins for continuous ones."""
    s = pd.Series(y)
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > bins:
        return pd.qcut(s.rank(method="first"), bins, labels=False, duplicates="drop").astype(int).to_numpy()
    return s.astype("category").cat.codes.to_numpy()


def parse_mofid(s):
    if not isinstance(s, str) or s.strip().lower() in BAD:
        return None
    body = s.split(";")[0].strip()
    m = _MOFID.search(body)
    if not m:
        return None
    smiles = body[:m.start()].strip()
    if not smiles or smiles.lower() in BAD:
        return None

    metals, linkers = set(), []
    for comp in smiles.split("."):
        if not comp:
            continue
        found = {e for inner in _BRACKET.findall(comp)
                 for e in _ELEMENT.findall(inner.split("_")[0]) if e in METALS}
        if found:
            metals |= found
        else:
            linkers.append(comp)

    toks = [t.strip().lower() for t in m.group(1).split(".")[0].split(",") if t.strip()]
    topo = "*" if (not toks or any(t in BAD for t in toks)) else ",".join(sorted(set(toks)))
    return metals, ".".join(sorted(linkers)), topo


class DSU:

    def __init__(self):
        self.p: dict = {}

    def find(self, x):
        self.p.setdefault(x, x)
        r = x
        while self.p[r] != r:
            r = self.p[r]
        while self.p[x] != r:
            self.p[x], x = r, self.p[x]
        return r

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def mofid_terminal_id(s):
    if not isinstance(s, str) or s.strip().lower() in BAD:
        return None
    return s.rsplit(";", 1)[-1].strip() if ";" in s else None


def build_groups(df: pd.DataFrame, id_col, v1_col, v2_col, metal_col_name, refcode_col=None):
    """Union-find duplicate grouping on the metal|linker|topology key from MOFid v1/v2."""
    ids = df[id_col].astype(str).tolist()
    metal_col = df[metal_col_name].map(
        lambda v: ",".join(sorted({x.strip() for x in str(v).replace(",", " ").split()
                                   if x.strip()})))
    P = {tag: [parse_mofid(v) for v in (df[col] if col else [None] * len(df))]
         for tag, col in (("v1", v1_col), ("v2", v2_col))}

    _mis = {"v1": set(), "v2": set()}
    if refcode_col and refcode_col in df.columns:
        for tag, col in (("v1", v1_col), ("v2", v2_col)):
            if not col:
                continue
            for idx, (v, rc) in enumerate(zip(df[col], df[refcode_col])):
                t = mofid_terminal_id(v)
                if t is not None and str(rc).strip() and t != str(rc).strip():
                    _mis[tag].add(idx)

    topo, n_conflict, n_fill = [], 0, 0
    for i, (a, b) in enumerate(zip(P["v1"], P["v2"])):
        t1 = a[2] if (a and i not in _mis["v1"]) else "*"
        t2 = b[2] if (b and i not in _mis["v2"]) else "*"
        if t1 != "*" and t2 != "*" and t1 != t2:
            n_conflict += 1
        if t1 == "*" and t2 != "*":
            n_fill += 1
        topo.append(t1 if t1 != "*" else t2)

    misaligned = _mis

    st = dict(v1_key=0, v2_key=0, contaminated=0, pillar=0, misaligned=0,
              topo_conflict=n_conflict, topo_filled=n_fill, topo_unknown=0)

    dsu = DSU()
    for i in ids:
        dsu.find(("S", i))

    contaminated, pillar_rows = [], []
    for tag in ("v1", "v2"):
        for idx, (sid, r) in enumerate(zip(ids, P[tag])):
            if r is None:
                continue
            key_metals, linker, _ = r
            if idx in misaligned[tag]:
                st["misaligned"] += 1
                contaminated.append((sid, tag, "TERMINAL_ID_MISMATCH",
                                     str(df[refcode_col].iloc[idx])))
                continue
            col_m = set(metal_col.iloc[idx].split(",")) - {""}

            if key_metals and col_m and not (key_metals <= col_m):
                st["contaminated"] += 1
                contaminated.append((sid, tag, ",".join(sorted(key_metals)),
                                     ",".join(sorted(col_m))))
                continue
            if key_metals and col_m and key_metals < col_m:
                st["pillar"] += 1
                pillar_rows.append((sid, tag, ",".join(sorted(key_metals)),
                                    ",".join(sorted(col_m))))
            if topo[idx] == "*":
                continue

            st[f"{tag}_key"] += 1
            dsu.union(("K", f"{tag}|{metal_col.iloc[idx]}|{linker}|{topo[idx]}"),
                      ("S", sid))

    st["topo_unknown"] = sum(1 for t in topo if t == "*")

    roots = {s: dsu.find(("S", s)) for s in ids}
    order = {r: i for i, r in enumerate(sorted(set(roots.values()), key=str))}
    g = pd.Series([order[roots[s]] for s in ids], index=df.index)

    cover = []
    for i, (a, b) in enumerate(zip(P["v1"], P["v2"])):
        if a is None and b is None:
            cover.append("no_MOFid")
        elif topo[i] == "*":
            cover.append("topology_unknown")
        else:
            cover.append("full_key")
    return g, pd.Series(cover, index=df.index), st, metal_col, contaminated, pillar_rows



def stratified_group_split(strata, groups, n_folds, seed):
    """Fold k = n_folds-2 -> val, n_folds-1 -> test, the rest train (n_folds=10 gives 8:1:1)."""
    from sklearn.model_selection import StratifiedGroupKFold
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = np.empty(len(strata), dtype=int)
    for k, (_, idx) in enumerate(skf.split(np.zeros(len(strata)), strata, groups)):
        fold[idx] = k
    return np.where(fold <= n_folds - 3, "train", np.where(fold == n_folds - 2, "val", "test"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", required=True,
                   help="label CSV: id column + 'label' (classification) or id column + target columns (regression)")
    p.add_argument("--task", choices=["classification", "regression"], default="classification")
    p.add_argument("--outdir", default="data/split", help="output folder (default: data/split)")
    p.add_argument("--id_col", default="id", help="id column in the label file and --structures (default: id)")
    p.add_argument("--structures", default=None,
                   help="optional CSV with one row per structure holding the grouping columns below")
    p.add_argument("--group_col", default=None,
                   help="column of --structures with an existing group id (used as is)")
    p.add_argument("--mofid_v1_col", default=None, help="column with MOFid-v1 strings")
    p.add_argument("--mofid_v2_col", default=None, help="column with MOFid-v2 strings")
    p.add_argument("--metal_col", default=None, help="column with the metal types (required with MOFid columns)")
    p.add_argument("--refcode_col", default=None, help="optional column cross-checked against the MOFid ending")
    p.add_argument("--n_folds", type=int, default=10,
                   help="number of folds; one validation fold, one test fold, the rest train (default: 10, i.e. 8:1:1)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        lines.append(s)

    lab = pd.read_csv(args.labels)
    if args.id_col not in lab.columns:
        sys.exit(f"id column '{args.id_col}' not in {args.labels}: {list(lab.columns)}")
    lab[args.id_col] = lab[args.id_col].astype(str)
    if lab[args.id_col].duplicated().any():
        sys.exit("duplicated ids in the label file")
    target_cols = [c for c in lab.columns if c != args.id_col]
    df = lab[[args.id_col]].copy()
    df.columns = ["id"]

    mofid = args.mofid_v1_col or args.mofid_v2_col
    if args.group_col and mofid:
        sys.exit("use either --group_col or the MOFid columns, not both")
    if args.group_col or mofid:
        if not args.structures:
            sys.exit("--structures is required with --group_col or the MOFid columns")
        st = pd.read_csv(args.structures)
        st[args.id_col] = st[args.id_col].astype(str)
        cols = [c for c in (args.group_col, args.mofid_v1_col, args.mofid_v2_col,
                            args.metal_col, args.refcode_col) if c]
        missing = [c for c in cols if c not in st.columns]
        if missing:
            sys.exit(f"columns {missing} not in {args.structures}: {list(st.columns)}")
        st = st.drop_duplicates(args.id_col).set_index(args.id_col)
        absent = [i for i in df["id"] if i not in st.index]
        if absent:
            sys.exit(f"{len(absent)} ids of the label file are not in {args.structures}, e.g. {absent[:3]}")
        st = st.loc[df["id"]].reset_index().rename(columns={args.id_col: "id"})
    if args.group_col:
        df["group_id"] = pd.factorize(st[args.group_col])[0]
        log(f"groups from column '{args.group_col}'")
    elif mofid:
        if not args.metal_col:
            sys.exit("--metal_col is required with the MOFid columns")
        g, cover, stats, _, _, _ = build_groups(
            st, "id", args.mofid_v1_col, args.mofid_v2_col, args.metal_col, args.refcode_col)
        df["group_id"] = g.to_numpy()
        log("groups from MOFid (metal | linker | topology)")
        log(f"  structures without a usable key stay singletons: {int((cover != 'full_key').sum())}")
    else:
        df["group_id"] = np.arange(len(df))
        log("no grouping given: every structure is its own group")

    if args.task == "classification":
        if "label" not in lab.columns:
            sys.exit(f"classification needs a 'label' column; found {list(lab.columns)}")
        strata = make_strata(lab["label"], 10)
    else:
        strata = make_strata(lab[target_cols[0]], 10)
    n_groups = df["group_id"].nunique()
    if n_groups < args.n_folds:
        sys.exit(f"{n_groups} groups are fewer than --n_folds {args.n_folds}; lower --n_folds")
    df["split"] = stratified_group_split(strata, df["group_id"].to_numpy(), args.n_folds, args.seed)
    assert (df.groupby("group_id")["split"].nunique() == 1).all(), "a group spans several subsets"

    log(f"\nstructures {len(df)}   groups {n_groups}")
    log(f"{'split':8s} {'structures':>11s} {'groups':>8s}" + ("   class counts" if args.task == "classification" else ""))
    for s in ("train", "val", "test"):
        sub = df["split"] == s
        extra = ""
        if args.task == "classification":
            extra = "   " + str(lab.loc[sub, "label"].value_counts().sort_index().to_dict())
        log(f"{s:8s} {int(sub.sum()):11d} {df.loc[sub, 'group_id'].nunique():8d}{extra}")

    js = {s: sorted(df.loc[df["split"] == s, "id"]) for s in ("train", "val", "test")}
    meta = dict(seed=args.seed, n_folds=args.n_folds, task=args.task, n_structures=len(df),
                n_groups=int(n_groups), counts={k: len(v) for k, v in js.items()})
    (outdir / "split.json").write_text(json.dumps({**meta, **js}, indent=1), encoding="utf-8")
    df.to_csv(outdir / "split_assignment.csv", index=False)
    grp = df.groupby("group_id").agg(n=("id", "size"), split=("split", "first"),
                                     ids=("id", lambda s: ";".join(s)))
    grp[grp["n"] > 1].sort_values("n", ascending=False).to_csv(outdir / "duplicate_groups.csv")
    (outdir / "split_log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"\nSaved split.json, split_assignment.csv, duplicate_groups.csv, split_log.txt to {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
