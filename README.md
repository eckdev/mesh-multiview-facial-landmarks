# Mesh and multi-view 3D facial landmark localization

Code for the model described in *A coarse-to-fine hybrid mesh and multi-view deep
learning model for 3D facial landmark localization* (Kayadarçın and Kurt, 2026).

The model localizes 23 facial landmarks on textured 3D surface scans in four stages:

1. **Stage 1:** global coarse localization from the whole surface.
2. **Stage 2:** refinement with attention between anatomically related landmarks
   and a geodesic candidate region.
3. **Shape prior:** a Procrustes shape prior that adjusts the configuration as a whole.
4. **Multi-view refiner:** scores rendered image evidence (three landmark-centred
   views and one context view) and geometric features on shared 3D candidate
   points. Six refiners are averaged.

## Repository layout

| Path | Contents |
|---|---|
| `facial_landmarks/core/` | Data loading, rigid alignment, Stage 1, training loop, metrics, cross-validation splits |
| `facial_landmarks/stage2/` | Stage 2 network, test-time augmentation and its runner |
| `facial_landmarks/refiner/` | Multi-view refiner, Procrustes shape prior and spatial prior |
| `facial_landmarks/refiner_base/` | Image encoder and view rendering shared with the refiner |
| `facial_landmarks/optional/` | Two refiners that are **disabled in the final model**. Stage 2 imports them, so they are included. |
| `facial_landmarks/pipeline/` | Five-fold cross-validation of the final model (`run_cv.py`), Colab presets (`colab_run.py`), component analysis (`run_ablation.py`), cost measurement (`measure_cost.py`) |
| `configs/` | Configuration and decision record of the reported run; fold assignment (scan IDs only) |

## Installation

```bash
pip install -r requirements.txt
python -m pytest            # 145 unit tests, CPU only
```

## Data

The scans and reference landmarks are **not** included; they cannot be shared.
The code expects:

- one coloured triangular mesh per participant in PLY format;
- one text file per participant with the 23 landmarks in the order listed in
  `facial_landmarks/core/anatomy.py`, using the layout read by
  `facial_landmarks.core.data.read_landmarks`.

The fold assignment of the reported experiments is given in
`configs/folds_5fold_192_48_60_seed42.json`.

## Running the cross-validation

The experiments were run on Google Colab with an NVIDIA A100 GPU. Paths to the
data and to earlier stage outputs are set at the top of
`facial_landmarks/stage2/colab.py` and `facial_landmarks/pipeline/colab_run.py`.

1. **Stages 1 and 2** run with `facial_landmarks/stage2/runner.py`. The preset
   used for the publication runs is in `facial_landmarks/stage2/colab.py`. It
   expects a preprocessing root (rigid alignment, mesh cache, Stage 1
   out-of-fold estimates) produced by `facial_landmarks/core/runner.py`.
2. **Shape prior and refiner** run with `facial_landmarks/pipeline/run_cv.py`.
   The final model corresponds to the `v2final_meshonly_*` presets of
   `facial_landmarks/pipeline/colab_run.py`. For example:

   ```bash
   python facial_landmarks/pipeline/colab_run.py --preset v2final_meshonly_val_all_folds
   python facial_landmarks/pipeline/colab_run.py --preset v2final_meshonly_final_test
   ```

`measure_cost.py` also contains the timing code for the three comparison
networks. That code needs their separate implementations, which are not part of
this repository. Use `--models proposed` to time the proposed model only.

## Names kept from development

The packages were renamed for this release. Class names (for example
`AGHFormerVNext`, `All23MVSCNet`), command-line options (`--all23-*`), preset
names and output folder names (`fold_k/all23_mvsc`) were kept, because they are
part of the stored configurations and configuration signatures. Changing them
would make the trained models unusable.

`facial_landmarks/_checkpoint_compat.py` maps the original package names to the
renamed modules, so that checkpoints saved before the renaming still load.

## Trained models

The trained weights are not included in this repository.

## Citation

If you use this code, please cite the article above.
