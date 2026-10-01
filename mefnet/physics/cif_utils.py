"""Read a CIF structure and its per-site partial charges."""
from __future__ import annotations

import numpy as np
from pymatgen.io.cif import CifParser

_NO_CHARGE_MSG = ("no per-atom partial charges (_atom_site_charge) found. MEFNet needs CIFs with "
                  "partial charges; assign them first, e.g. with PACMAN.")


def load_structure(cif_path: str):
    parser = CifParser(cif_path)
    try:
        structures = parser.parse_structures(primitive=False)
    except AttributeError:
        structures = parser.get_structures(primitive=False)
    if not structures:
        raise ValueError(f"No structure parsed from {cif_path}")
    return parser, structures[0]


def _first_block(parser: CifParser):
    data = parser._cif.data
    if not data:
        raise ValueError("Empty CIF data block")
    return next(iter(data.values()))


def extract_cif_charges(parser: CifParser) -> np.ndarray | None:
    block = _first_block(parser)
    for key in ("_atom_site_charge", "_atom_site_partial_charge", "_atom_site_net_charge"):
        if key in block.data:
            raw = block.data[key]
            try:
                return np.asarray([float(x) for x in raw], dtype=np.float32)
            except Exception as exc:
                raise ValueError(f"Failed to parse CIF charges from {key}: {exc}") from exc
    return None


def load_structure_and_charges(
    cif_path: str,
    charge_path: str | None = None,
    charge_source: str = "auto",
):
    """Load a structure and charges from the CIF or a .npy file."""
    parser, structure = load_structure(cif_path)

    cif_charges = None
    if charge_source in ("auto", "cif"):
        cif_charges = extract_cif_charges(parser)

    npy_charges = None
    if charge_source in ("auto", "npy") and charge_path is not None:
        npy_charges = np.load(charge_path).astype(np.float32)

    charges = None
    source_used = None

    if charge_source == "cif":
        if cif_charges is None:
            raise ValueError(_NO_CHARGE_MSG)
        charges = cif_charges
        source_used = "cif"
    elif charge_source == "npy":
        if npy_charges is None:
            raise ValueError("NPY charges requested but not found")
        charges = npy_charges
        source_used = "npy"
    else:
        if cif_charges is not None:
            charges = cif_charges
            source_used = "cif"
        elif npy_charges is not None:
            charges = npy_charges
            source_used = "npy"
        else:
            raise ValueError(f"{cif_path}: {_NO_CHARGE_MSG}")

    if len(charges) != len(structure):
        raise ValueError(
            f"Charge length mismatch for {cif_path}: "
            f"{len(charges)} charges vs {len(structure)} structure sites"
        )

    return structure, charges, source_used
