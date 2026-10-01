"""Build charged crystal graphs (.npz) from CIF files with partial charges."""
from __future__ import annotations
import json
import os
import warnings
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifParser


class GaussianDistance:
    def __init__(self, dmin, dmax, step, var=None):
        self.filter = np.arange(dmin, dmax+step, step)
        self.var = var if var else step

    def expand(self, distances):
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)


class AtomInitializer:
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        self._embedding = {int(k): np.array(v, dtype=np.float32)
                          for k, v in elem_embedding.items()}

    def get_atom_fea(self, atom_number):
        return self._embedding[atom_number]


def get_charges_from_cif(cif_path, n_atoms, strict=True):
    """Read _atom_site_charge from the CIF; fail if missing unless strict=False."""
    try:
        parser = CifParser(cif_path)
        data = list(parser.as_dict().values())[0]
        if '_atom_site_charge' not in data:
            raise KeyError('_atom_site_charge')
        charges = [float(c) for c in data['_atom_site_charge']]
    except Exception as exc:
        msg = (f'{cif_path}: no per-atom partial charges (_atom_site_charge) could be read ({exc}). '
               'MEFNet needs CIFs with partial charges; assign them first, e.g. with PACMAN.')
        if strict:
            raise ValueError(msg) from exc
        warnings.warn(msg + ' Using zeros.')
        return np.zeros(n_atoms, dtype=np.float32)

    if len(charges) != n_atoms:
        msg = (f'{cif_path}: {len(charges)} charges vs {n_atoms} atoms. '
               'A primitive reduction or a site merge is the usual cause.')
        if strict:
            raise ValueError(msg)
        warnings.warn(msg + ' Using zeros.')
        return np.zeros(n_atoms, dtype=np.float32)
    return np.array(charges, dtype=np.float32)


def process_cif(cif_path, ari, gdf, max_num_nbr=12, radius=8, strict=True):
    """Convert one CIF into graph arrays (atom features, charges, neighbours, volume)."""
    structure = Structure.from_file(cif_path, primitive=False)
    n_atoms = len(structure)

    atom_fea = np.vstack([
        ari.get_atom_fea(structure[i].specie.number)
        for i in range(n_atoms)
    ]).astype(np.float32)

    charges = get_charges_from_cif(cif_path, n_atoms, strict=strict)

    all_nbrs = structure.get_all_neighbors(radius, include_index=True)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]

    nbr_fea_idx, nbr_fea = [], []
    for nbr in all_nbrs:
        if len(nbr) < max_num_nbr:
            nbr_fea_idx.append(
                list(map(lambda x: x[2], nbr)) +
                [0] * (max_num_nbr - len(nbr))
            )
            nbr_fea.append(
                list(map(lambda x: x[1], nbr)) +
                [radius + 1.] * (max_num_nbr - len(nbr))
            )
        else:
            nbr_fea_idx.append(list(map(lambda x: x[2], nbr[:max_num_nbr])))
            nbr_fea.append(list(map(lambda x: x[1], nbr[:max_num_nbr])))

    nbr_fea = gdf.expand(np.array(nbr_fea))
    nbr_fea_idx = np.array(nbr_fea_idx)
    volume = float(structure.volume)

    net_q = float(charges.sum())
    if abs(net_q) > 0.05:
        warnings.warn(f'{cif_path}: net charge {net_q:+.4f} is not neutral')

    return {
        'atom_fea': atom_fea,
        'charges': charges,
        'nbr_fea': nbr_fea.astype(np.float32),
        'nbr_fea_idx': nbr_fea_idx.astype(np.int64),
        'volume': np.float32(volume),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cif_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--atom_init', type=str,
                        default=str(Path(__file__).parent / 'mefnet' / 'data' / 'atom_init.json'),
                        help='element embedding JSON (default: bundled atom_init.json)')
    parser.add_argument('--max_num_nbr', type=int, default=12)
    parser.add_argument('--radius', type=float, default=8.0)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--allow_missing_charge', action='store_true',
                        help='fill zeros when charges cannot be read (default: fail)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ari = AtomInitializer(args.atom_init)
    gdf = GaussianDistance(dmin=0, dmax=args.radius, step=0.2)

    cif_files = sorted(Path(args.cif_dir).glob('*.cif'))
    if args.limit:
        cif_files = cif_files[:args.limit]

    success, failed = [], []
    for cif_path in tqdm(cif_files, desc="Generating graphs"):
        mof_id = cif_path.stem
        out_path = output_dir / f'{mof_id}.npz'
        try:
            data = process_cif(str(cif_path), ari, gdf,
                             args.max_num_nbr, args.radius,
                             strict=not args.allow_missing_charge)
            np.savez(out_path, **data)
            success.append(mof_id)
        except Exception as e:
            failed.append({'mof_id': mof_id, 'error': str(e)})

    print(f"Done. Success: {len(success)}, Failed: {len(failed)}")
    if failed:
        print("Failed structures:")
        for f in failed:
            print(f"  {f['mof_id']}: {f['error']}")
        raise SystemExit(1)


if __name__ == '__main__':
    main()
