# Examples

Twenty MOFs taken from the CoRE MOF 2024 database (https://zenodo.org/records/15055758). All of them come from the
database's literature-SI subset. The CIFs contain DDEC6 partial charges (`_atom_site_charge`) computed with PACMAN.

| File | Content |
|---|---|
| `cifs/` | 20 CIFs, 34-132 atoms each |
| `labels_classification.csv` | `id`, `label` (1 = hydrophilic, 0 = hydrophobic); 10 structures per class |
| `labels_regression.csv` | `id`, `log_KH` (log10 of the Henry coefficient of water, mmol/g/Pa) |
| `structures.csv` | `id`, `refcode`, `mofid-v1`, `mofid-v2`, `metal_types`, `natoms`, `source` |

Four pairs of structures share a MOFid-based group (duplicate pairs), so the group split keeps them in the same subset.
