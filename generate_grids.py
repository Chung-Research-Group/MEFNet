#!/usr/bin/env python3
"""Generate Ewald electrostatic grids (.npy) from CIF files in parallel."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def _pin_blas_threads():
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"


_pin_blas_threads()


def _worker(task):
    """Generate the grid for one structure."""
    cif_path, out_path, meta_path, cfg_kwargs = task
    _pin_blas_threads()
    try:
        from mefnet.physics.ewald_grid_generator import EwaldGridConfig, generate_grid_from_files
        t0 = time.time()
        cfg = EwaldGridConfig(**cfg_kwargs)
        generate_grid_from_files(
            cif_path=cif_path,
            output_npy=out_path,
            config=cfg,
            output_meta_json=meta_path,
        )
        return (Path(cif_path).stem, True, time.time() - t0, "")
    except Exception as exc:
        return (Path(cif_path).stem, False, 0.0, f"{type(exc).__name__}: {exc}")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__)
    p.add_argument("--cif_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--workers", type=int, default=0,
                   help="number of worker processes; 0 = 3/4 of cores")
    p.add_argument("--benchmark", type=int, default=0,
                   help="run only N structures and estimate the total runtime")
    p.add_argument("--resume", action="store_true",
                   help="skip structures whose .npy already exists")
    p.add_argument("--report", default=None, help="path of the report JSON")
    p.add_argument("--grid_size", type=int, default=30)
    p.add_argument("--alpha", type=float, default=0.25)
    p.add_argument("--real_nmax", type=int, default=1)
    p.add_argument("--recip_hmax", default="auto",
                   help='reciprocal-space cutoff: an integer or "auto"\n'
                        '  auto: per-axis cutoff from the cell size (RASPA2 rule, recommended)\n'
                        '  4   : fixed for every cell; too small for large cells\n'
                        '')
    p.add_argument("--chunk_size", type=int, default=512)
    p.add_argument("--no_normalize", action="store_true")
    args = p.parse_args()

    cif_dir = Path(args.cif_dir)
    out_dir = Path(args.output_dir)
    meta_dir = out_dir / "_meta"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    cifs = sorted(cif_dir.glob("*.cif"))
    if not cifs:
        print(f"[STOP] no CIF in {cif_dir}")
        return 2

    hmax = args.recip_hmax
    if isinstance(hmax, str) and hmax.lower() != "auto":
        hmax = int(hmax)

    cfg_kwargs = dict(grid_size=args.grid_size, alpha=args.alpha,
                      real_nmax=args.real_nmax, recip_hmax=hmax,
                      chunk_size=args.chunk_size,
                      normalize=not args.no_normalize)

    cifs.sort(key=lambda c: c.stat().st_size, reverse=True)

    if args.resume:
        before = len(cifs)
        cifs = [c for c in cifs if not (out_dir / f"{c.stem}.npy").exists()]
        print(f"  --resume : skipped {before - len(cifs)}, {len(cifs)} remaining")

    if args.benchmark:
        cifs = cifs[: args.benchmark]

    workers = args.workers or max(1, (os.cpu_count() or 4) * 3 // 4)
    print("=" * 62)
    print(f"CIF {len(cifs)}   workers {workers}   cores {os.cpu_count()} available")
    print(f"grid_size={args.grid_size} alpha={args.alpha} "
          f"real_nmax={args.real_nmax} recip_hmax={args.recip_hmax}")
    print("=" * 62)

    tasks = [(str(c), str(out_dir / f"{c.stem}.npy"),
              str(meta_dir / f"{c.stem}.json"), cfg_kwargs) for c in cifs]

    ok, failed, times = [], [], []
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_worker, t): t[0] for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            name, good, dt, err = fut.result()
            if good:
                ok.append(name)
                times.append(dt)
            else:
                failed.append({"mof_id": name, "error": err})
                print(f"  [FAIL] {name}: {err}")
            if i % 25 == 0 or i == len(tasks):
                elapsed = time.time() - t_start
                rate = i / max(elapsed, 1e-9)
                remain = (len(tasks) - i) / max(rate, 1e-9)
                print(f"  {i}/{len(tasks)}  elapsed {elapsed/60:.1f} min  "
                      f"remaining {remain/60:.1f} min  "
                      f"({rate*3600:.0f} per hour)")

    total = time.time() - t_start
    print()
    print("=" * 62)
    print(f"success {len(ok)}   failed {len(failed)}   total {total/60:.1f} min")
    if times:
        times_sorted = sorted(times)
        print(f"time per structure: median {times_sorted[len(times)//2]:.1f}s   "
              f"max {times_sorted[-1]:.1f}s   sum {sum(times)/3600:.1f} h (CPU)")
        print(f"parallel speed-up: {sum(times)/max(total,1e-9):.1f}x  (with {workers} workers)")

    if args.benchmark:
        n_all = len(sorted(cif_dir.glob('*.cif')))
        per = total / max(len(ok), 1)
        print()
        print(f"estimated time for all {n_all}: {n_all*per/3600:.1f} h  (with {workers} workers)")
        if n_all * per / 3600 > 24:
            need = int(workers * (n_all * per / 3600) / 20)
            print(f"  Exceeds 24 h. Use at least {need} workers,")
            print(f"  lower --recip_hmax, or split the run across nodes.")
    print("=" * 62)

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(
            {"success": ok, "failed": failed, "config": cfg_kwargs,
             "workers": workers, "total_seconds": total},
            ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"report: {args.report}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
