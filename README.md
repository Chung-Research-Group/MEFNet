<p align="center">
  <img src="assets/MEFNet_banner.png" alt="MEFNet: Multi-modal Electrostatic Field Network" width="100%">
</p>

# MEFNet

MEFNet predicts MOF properties from three inputs computed for each structure:
a charge-embedded crystal graph, a 3D Ewald electrostatic grid (potential and field components) and the unit cell volume.
It was developed for hydrophobicity classification and also supports regression.

The graph encoder is a CGCNN-style network with per-atom partial-charge embeddings. The grid is encoded by a patch-based 3D
network. Three ways of pooling the grid tokens are available (`--fusion_type`): `none` (mean pooling, the default and the final
model of the paper), `gate` (graph-gated) and `gcsa` (graph-conditioned cross-attention, GCCA).

## Install

```bash
conda create -n mefnet python=3.11
conda activate mefnet
git clone https://github.com/yoonseonghyun/MEFNet.git && cd MEFNet
pip install -r requirements.txt
```

Install PyTorch separately by following https://pytorch.org/get-started/locally/ for your CUDA version. The weights and the
examples were run with PyTorch 2.6.0 and Python 3.11. A GPU is optional.

## Input data

| File | Required columns / content | Used by |
|---|---|---|
| CIF files, one per structure (`<id>.cif`) | Atom sites with partial charges in `_atom_site_charge`, one per site | `generate_graphs.py`, `generate_grids.py` |
| Label CSV | `id` plus `label` (classification, integer 0/1), or `id` plus one or more numeric target columns (regression; every non-id column is a target) | `split_*.py`, `train.py`, `tune.py`, `predict.py` |
| Structure CSV (optional) | `id` plus the grouping columns below | `split_group.py` |

The id column of the label CSV may be named `id`, `mof_id`, `name` or `coreid`. Ids must equal the CIF file names without `.cif`.

**Partial charges.** MEFNet needs a charge for every atom. A CIF without `_atom_site_charge` makes `generate_graphs.py` and
`generate_grids.py` stop with an error. Assign charges first, for example with PACMAN
(https://github.com/mtap-research/PACMAN-charge), which writes `_atom_site_charge`.

**Groups (optional).** Structures that are near-duplicates (same metal, linkers and topology) should not be split across train and
test. `split_group.py` accepts one of:

| Option | Meaning | If absent |
|---|---|---|
| `--group_col` | Column of the structure CSV with an existing group id | - |
| `--mofid_v1_col` / `--mofid_v2_col` + `--metal_col` | MOFid-v1 / v2 strings and the metal types; groups are built from metal, linker and topology (the metal column is required with them; `--refcode_col` is an optional cross-check) | - |
| none of the above | Every structure is its own group, which is equivalent to a random split | default |

**Folder structure** used by the commands below (all paths can be changed with options):

```
data/graphs/        <id>.npz            from generate_graphs.py
data/grids/         <id>.npy            from generate_grids.py
data/split/         split.json, split_assignment.csv
labels.csv          your label file
```

`data/coremof2024/` holds the dataset and splits used in the paper: the binary hydrophobicity labels, the group split
(`split.json`, `split_assignment.csv`) and the random split (`split_random.json`, `split_assignment_nogroup.csv`).
CIFs are available from the CoRE MOF 2024 database (https://zenodo.org/records/15055758).

`examples/` contains 20 CoRE MOF 2024 structures with classification and regression labels (see `examples/README.md`).

## Graphs and grids

```bash
python generate_graphs.py --cif_dir examples/cifs --output_dir data/graphs
python generate_grids.py  --cif_dir examples/cifs --output_dir data/grids --workers 4
```

Graph settings: cutoff radius 8 Å, 12 neighbours, Gaussian distance expansion with 0.2 Å step (`--radius`, `--max_num_nbr`).
Grid settings: 30 x 30 x 30 points, Ewald splitting parameter 0.25 Å⁻¹, four channels (potential and three field components),
z-score normalised (`--grid_size`, `--alpha`, `--recip_hmax`, `--no_normalize`). `--resume` skips existing grids. A grid takes
seconds to minutes per structure depending on the cell size, so use `--workers` for large sets.

## Split

Group split (default 8:1:1; `--n_folds` sets the number of folds, one validation fold, one test fold, the rest train):

```bash
python split_group.py --labels examples/labels_classification.csv --structures examples/structures.csv \
    --mofid_v1_col mofid-v1 --mofid_v2_col mofid-v2 --metal_col metal_types \
    --n_folds 5 --outdir data/split
```

Classification splits are stratified by label; regression splits are stratified by quantile bins of the first target.
`--task regression` selects the latter. Outputs: `split.json`, `split_assignment.csv` (`id`, `group_id`, `split`),
`duplicate_groups.csv`, `split_log.txt`.

Random split without grouping:

```bash
python split_random.py --labels examples/labels_classification.csv --outdir data/split_random
```

`--val_frac` and `--test_frac` set the sizes (default 0.1 each).

## Training

```bash
# classification (default fusion_type: none)
python train.py --run_name cls --label_file examples/labels_classification.csv \
    --split_file data/split/split.json --group_file data/split/split_assignment.csv \
    --graph_dir data/graphs --grid_dir data/grids

# regression
python train.py --run_name reg --task regression --no_softmax \
    --label_file examples/labels_regression.csv --split_file data/split/split.json \
    --group_file data/split/split_assignment.csv

# other grid pooling: --fusion_type gate | gcsa | none
```

`--no_softmax` is required for scalar regression targets. Without it the outputs are passed through a softmax and the targets are
treated as a probability distribution. `--group_file` is optional; without it every structure is its own group.

Training saves `best.pt` (best validation score: material-level AUC, or negative material-level MAE), `config.json`,
`history.json`, `results.json` and `test_predictions.csv` (classification) with a confusion matrix to
`results/<run_name>/`. Metrics are reported per structure and per material, where a material is the average over the
structures of one (group, label) pair. If the test set contains only one class, metrics are not defined and only predictions are saved.
Class weights for the loss are computed from the training labels, and every sample is additionally weighted by the inverse size of
its (group, label) unit.

Other options: `--max_epochs`, `--batch_size`, `--patience`, `--seed`, `--no_grid`, `--no_graph`, `--no_volume`, `--no_charge`,
`--result_dir`.

## Hyperparameter tuning

```bash
python tune.py --output results/tune --n_trials 100 \
    --label_file examples/labels_classification.csv --split_file data/split/split.json \
    --group_file data/split/split_assignment.csv

python train.py --run_name tuned --params_json results/tune/best_params.json \
    --label_file examples/labels_classification.csv --split_file data/split/split.json \
    --group_file data/split/split_assignment.csv
```

`tune.py` runs an Optuna search (TPE, seed 42) maximising the validation material-level AUC. With `--task regression` it minimises
the validation material-level MAE (pass `--no_softmax` for scalar targets). It takes the same data and model options as `train.py`
(`--fusion_type`, `--no_grid`, ...), `--max_epochs` (default 100) and `--patience` (default 10) per trial. The search space is
`SEARCH_SPACE` in `tune.py`. Outputs: `best_params.json`, `all_trials.csv`, `param_importances.json`.
`--params_json` loads `best_params` from such a file into `train.py`.

## Evaluation and prediction

Weights trained on the CoRE MOF 2024 hydrophobicity dataset (seed 42) are in `weights/`:

| Folder | `--fusion_type` | Pooling |
|---|---|---|
| `weights/mean` | `none` | Mean (final model) |
| `weights/gate` | `gate` | Gate |
| `weights/gcca` | `gcsa` | GCCA |

Each folder contains `best.pt` and `config.json`. A run folder written by `train.py` can be used in the same way.

```bash
# predict a subset of a split and evaluate it
python predict.py --model_dir weights/mean --graph_dir data/graphs --grid_dir data/grids \
    --split_file data/split/split.json --subset test \
    --label_file examples/labels_classification.csv --output results/pred.csv

# predict without labels (all structures with a graph and a grid in the folders)
python predict.py --model_dir weights/mean --graph_dir data/graphs --grid_dir data/grids --output results/pred.csv
```

The output CSV has `id`, `prob_class_1` and `pred_class` (classification) or `pred_<target>` columns (regression). With
`--label_file`, metrics are printed and written to `<output>_metrics.json`. `--group_file` enables material-level metrics for
duplicate groups. The provided weights are classifiers. The example structures belong to the dataset they were trained on, so metrics on them are not an independent test.

## Quick start

Runs on the examples in a few minutes, on a CPU.

```bash
python generate_graphs.py --cif_dir examples/cifs --output_dir data/graphs
python generate_grids.py  --cif_dir examples/cifs --output_dir data/grids --workers 4

python split_group.py --labels examples/labels_classification.csv --structures examples/structures.csv \
    --mofid_v1_col mofid-v1 --mofid_v2_col mofid-v2 --metal_col metal_types --n_folds 5 --outdir data/split

python tune.py --output results/tune --n_trials 3 --max_epochs 3 \
    --label_file examples/labels_classification.csv --split_file data/split/split.json \
    --group_file data/split/split_assignment.csv

python train.py --run_name cls --params_json results/tune/best_params.json --max_epochs 20 --patience 20 \
    --label_file examples/labels_classification.csv --split_file data/split/split.json \
    --group_file data/split/split_assignment.csv

python train.py --run_name reg --task regression --no_softmax --max_epochs 20 --patience 20 \
    --label_file examples/labels_regression.csv --split_file data/split/split.json \
    --group_file data/split/split_assignment.csv

python predict.py --model_dir results/cls --graph_dir data/graphs --grid_dir data/grids \
    --split_file data/split/split.json --subset test \
    --label_file examples/labels_classification.csv --output results/cls/test_pred.csv
```

With 20 structures the metrics are not meaningful; the example only demonstrates the workflow.

## Citation

If you use MEFNet, please cite:

S. Yoon and Y. G. Chung, "Integrating charge information into multi-modal learning for predicting metal-organic framework hydrophobicity", in preparation.

## References

- CGCNN (graph encoder, `atom_init.json`): T. Xie and J. C. Grossman, Phys. Rev. Lett. 2018, 120, 145301. https://doi.org/10.1103/PhysRevLett.120.145301
- CoRE MOF 2024 (examples and paper dataset): G. Zhao et al., Matter 2025, 8, 102140. https://doi.org/10.1016/j.matt.2025.102140
- PACMAN (partial charges): G. Zhao and Y. G. Chung, J. Chem. Theory Comput. 2024, 20, 5368-5380. https://doi.org/10.1021/acs.jctc.4c00434
- MOFid (grouping in `split_group.py`): B. J. Bucior et al., Cryst. Growth Des. 2019, 19, 6682-6697. https://doi.org/10.1021/acs.cgd.9b01050

## License

MIT License, see `LICENSE`. Parts adapted from CGCNN are covered by its MIT license, see `THIRD_PARTY_NOTICES`.
