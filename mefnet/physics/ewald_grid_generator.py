"""Ewald electrostatic potential and field grids (phi, Ex, Ey, Ez) on the unit cell."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import pi, sqrt
from pathlib import Path
import json
import itertools

import numpy as np
from scipy.special import erfc

from .cif_utils import load_structure_and_charges
from .lj_params import WATER_O_EPSILON_K, WATER_O_SIGMA, get_lj_params

COULOMB_CONSTANT_EV_A_PER_E2 = 14.3996454784255
SQRT_PI = sqrt(np.pi)


@dataclass
class EwaldGridConfig:
    """Settings for Ewald grid generation."""
    grid_size: int = 30
    alpha: float = 0.25
    real_nmax: int = 1
    recip_hmax: int | str = 4
    ewald_precision: float = 1.0e-6
    real_cutoff: float = 12.0
    include_lj: bool = False
    normalize: bool = True
    lj_clip_k: float = 1.0e5
    min_distance: float = 0.75
    chunk_size: int = 512
    charge_source: str = "auto"
    include_self_term: bool = False

    @property
    def channel_names(self):
        names = ["phi", "Ex", "Ey", "Ez"]
        if self.include_lj:
            names.append("LJ")
        return names


def _reciprocal_matrix(lattice) -> np.ndarray:
    recip = getattr(lattice, "reciprocal_lattice", None)
    if recip is not None:
        return np.asarray(recip.matrix, dtype=np.float64)
    A = np.asarray(lattice.matrix, dtype=np.float64)
    return 2.0 * np.pi * np.linalg.inv(A).T


def _fractional_grid_points(grid_size: int) -> np.ndarray:
    coords = (np.arange(grid_size, dtype=np.float32) + 0.5) / float(grid_size)
    fx, fy, fz = np.meshgrid(coords, coords, coords, indexing="ij")
    return np.stack([fx, fy, fz], axis=-1).reshape(-1, 3)


def _direct_image_shifts(nmax: int):
    return np.asarray(list(itertools.product(range(-nmax, nmax + 1), repeat=3)), dtype=np.float32)


def raspa_kvec(lengths, alpha: float, precision: float = 1.0e-6,
               rcut: float = 12.0):
    """Per-axis reciprocal cutoff following the RASPA2 rule."""
    tol = np.sqrt(abs(np.log(precision * rcut)))
    tol1 = np.sqrt(-np.log(precision * rcut * (2.0 * tol * alpha) ** 2))
    return [max(1, int(round(0.25 + float(L) * alpha * tol1 / np.pi)))
            for L in lengths]


def _reciprocal_vectors(recip_hmax) -> np.ndarray:
    if np.isscalar(recip_hmax):
        hx = hy = hz = int(recip_hmax)
    else:
        hx, hy, hz = (int(v) for v in recip_hmax)
    hkls = []
    for h in range(-hx, hx + 1):
        for k in range(-hy, hy + 1):
            for l in range(-hz, hz + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                hkls.append((h, k, l))
    return np.asarray(hkls, dtype=np.int32)


def _compute_lj_parameters(symbols: list[str]):
    sigma_fw = []
    epsilon_fw = []
    for sym in symbols:
        sigma, eps = get_lj_params(sym)
        sigma_fw.append(sigma)
        epsilon_fw.append(eps)
    sigma_fw = np.asarray(sigma_fw, dtype=np.float32)
    epsilon_fw = np.asarray(epsilon_fw, dtype=np.float32)
    sigma_ij = 0.5 * (sigma_fw + WATER_O_SIGMA)
    epsilon_ij = np.sqrt(epsilon_fw * WATER_O_EPSILON_K)
    return sigma_ij, epsilon_ij


def _zscore_channels(grid: np.ndarray):
    meta = {"mean": [], "std": []}
    out = grid.copy()
    for i in range(grid.shape[0]):
        mean = float(out[i].mean())
        std = float(out[i].std())
        if std < 1e-12:
            std = 1.0
        out[i] = (out[i] - mean) / std
        meta["mean"].append(mean)
        meta["std"].append(std)
    return out, meta


def generate_grid_from_structure(structure, charges: np.ndarray, config: EwaldGridConfig):
    """Compute the Ewald potential/field grid for one structure."""
    frac_atoms = np.asarray(structure.frac_coords, dtype=np.float32)
    lattice = np.asarray(structure.lattice.matrix, dtype=np.float32)
    recip_lattice = _reciprocal_matrix(structure.lattice)
    volume = float(structure.volume)
    symbols = [site.specie.symbol for site in structure]
    frac_points = _fractional_grid_points(config.grid_size)

    n_points = frac_points.shape[0]
    phi = np.zeros(n_points, dtype=np.float64)
    Ex = np.zeros(n_points, dtype=np.float64)
    Ey = np.zeros(n_points, dtype=np.float64)
    Ez = np.zeros(n_points, dtype=np.float64)
    LJ = np.zeros(n_points, dtype=np.float64) if config.include_lj else None

    if config.include_lj:
        sigma_ij, epsilon_ij = _compute_lj_parameters(symbols)

    shifts = _direct_image_shifts(config.real_nmax)
    if isinstance(config.recip_hmax, str) and config.recip_hmax.lower() == "auto":
        edge = np.linalg.norm(lattice, axis=1)
        hmax_used = raspa_kvec(edge, config.alpha,
                               config.ewald_precision, config.real_cutoff)
    else:
        hmax_used = int(config.recip_hmax)
    hkls = _reciprocal_vectors(hmax_used)
    kvecs = hkls @ recip_lattice
    k2 = np.sum(kvecs ** 2, axis=1)
    recip_pref = 4.0 * np.pi / volume * np.exp(-k2 / (4.0 * config.alpha ** 2)) / k2

    atom_cart_all = frac_atoms @ lattice
    kr_atom = atom_cart_all @ kvecs.T
    struct_factor = (charges[:, None] * np.exp(-1j * kr_atom)).sum(axis=0)

    for start in range(0, n_points, config.chunk_size):
        stop = min(start + config.chunk_size, n_points)
        fp = frac_points[start:stop]

        phi_real = np.zeros(fp.shape[0], dtype=np.float64)
        E_real = np.zeros((fp.shape[0], 3), dtype=np.float64)

        for shift in shifts:
            diff_frac = fp[:, None, :] - (frac_atoms[None, :, :] + shift[None, None, :])
            r_vec = diff_frac @ lattice
            r = np.linalg.norm(r_vec, axis=2)
            r_safe = np.maximum(r, config.min_distance)

            erfc_ar = erfc(config.alpha * r_safe)
            exp_ar = np.exp(-(config.alpha ** 2) * (r_safe ** 2))
            phi_term = erfc_ar / r_safe
            field_coeff = erfc_ar / (r_safe ** 3) + (2.0 * config.alpha / SQRT_PI) * exp_ar / (r_safe ** 2)

            q = charges[None, :]
            phi_real += np.sum(q * phi_term, axis=1)
            E_real += np.sum(q[:, :, None] * field_coeff[:, :, None] * r_vec, axis=1)

            if config.include_lj and np.all(shift == 0):
                sr6 = (sigma_ij[None, :] / r_safe) ** 6
                lj_chunk = np.sum(4.0 * epsilon_ij[None, :] * (sr6 ** 2 - sr6), axis=1)
                lj_chunk = np.clip(lj_chunk, -config.lj_clip_k, config.lj_clip_k)
                LJ[start:stop] = lj_chunk

        r_cart = fp @ lattice
        kr_pt = r_cart @ kvecs.T
        z = struct_factor[None, :] * np.exp(1j * kr_pt)
        phi_recip = (recip_pref[None, :] * z.real).sum(axis=1)

        E_recip = np.einsum('bh,hd->bd', recip_pref[None, :] * z.imag, kvecs)

        phi_total = COULOMB_CONSTANT_EV_A_PER_E2 * (phi_real + phi_recip)
        E_total = COULOMB_CONSTANT_EV_A_PER_E2 * (E_real + E_recip)

        phi[start:stop] = phi_total
        Ex[start:stop] = E_total[:, 0]
        Ey[start:stop] = E_total[:, 1]
        Ez[start:stop] = E_total[:, 2]

    channels = [phi, Ex, Ey, Ez]
    if config.include_lj:
        channels.append(LJ)
    grid = np.stack(channels, axis=0).reshape(len(channels), config.grid_size, config.grid_size, config.grid_size)

    metadata = {
        "grid_size": config.grid_size,
        "alpha": config.alpha,
        "real_nmax": config.real_nmax,
        "recip_hmax": config.recip_hmax,
        "recip_hmax_used": (list(hmax_used) if not np.isscalar(hmax_used)
                            else int(hmax_used)),
        "n_kvectors": int(len(hkls)),
        "include_lj": config.include_lj,
        "channel_names": config.channel_names,
        "n_atoms": len(structure),
        "volume": float(structure.volume),
        "method": "ewald",
        "include_self_term": config.include_self_term,
    }

    if config.normalize:
        grid, norm_meta = _zscore_channels(grid)
        metadata["normalization"] = norm_meta

    return grid.astype(np.float32), metadata


def generate_grid_from_files(
    cif_path: str,
    output_npy: str,
    charge_path: str | None = None,
    config: EwaldGridConfig | None = None,
    output_meta_json: str | None = None,
):
    """Read a CIF, compute its grid and save it as .npy."""
    config = config or EwaldGridConfig()
    structure, charges, source_used = load_structure_and_charges(
        cif_path=cif_path,
        charge_path=charge_path,
        charge_source=config.charge_source,
    )
    grid, meta = generate_grid_from_structure(structure, charges, config)
    meta["charge_source"] = source_used
    np.save(output_npy, grid)

    if output_meta_json is not None:
        with open(output_meta_json, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    return grid, meta
