# Source-package audit and disposition

This repository was reorganized from the supplied `Code&data` folder. The
source folder itself was not modified.

## Inventory reviewed

| Type | Count | Approximate size | Disposition |
|---|---:|---:|---|
| Python source | 21 | 0.14 MB | active modules retained; superseded modules excluded |
| PNG images | 57 | 448 MB | all retained and classified by scientific role |
| XLSX workbooks | 11 | 0.06 MB | all retained |
| AVI video | 1 | 25.8 MB | retained as the example input |
| PyTorch weights | 1 | 23.5 MB | retained with a repository-relative default path |
| Python bytecode | 62 | — | excluded; regenerated automatically |
| IDE metadata | 8 | — | excluded; machine-specific and not scientific data |

## Python-module decisions

The active public pipeline retains `Geoborehole.py`, all five processing-stage
modules, the evaluation and XLSX-export modules, the DeepLabV3+/MobileNetV2
implementation, and the fracture aperture/trace helper.

The following supplied files were excluded from the public execution tree
after dependency tracing showed that they are not imported by the complete
pipeline:

- `step/lib/fracture/distribution.py`
- `step/lib/fracture/fracture.py`
- `step/lib/fracture/geometry.py`
- `step/lib/fracture/polygon.py`
- `step/lib/fracture/visualization.py`
- `step/lib/image/cluster.py`
- `step/lib/image/recognize.py`

They are earlier library/prototype paths, overlap with the maintained stage
modules, and would introduce additional undocumented APIs. Excluding dead code
makes the published route unambiguous. The original files remain recoverable
from the authors' untouched source folder.

`step/Caculate_evaluation.py` was corrected and renamed to
`step/evaluation.py`. The original misspelling is not retained as a second copy.

## Portability corrections

- Replaced the author-specific model path with
  `step/lib/image/logs/ep300-loss0.031-val_loss0.053.pth`.
- Replaced author-specific example input/output paths with repository-relative
  paths and command-line options.
- Added automatic CPU fallback and an explicit `--device` option.
- Replaced a non-public workbook-export runtime with `openpyxl`.
- Converted Python comments to English and moved detailed method explanations
  into `docs/METHODOLOGY.md`.
- Removed `.idea`, `__pycache__`, and `.pyc` files.
- Added package initializers so imports do not depend on `sys.path` mutation or
  another installed package named `lib`.

## Image-data decisions

All 57 supplied PNG files are present. Seven case-study figures are under
`data/case_study_figures/`; the 50 full example pipeline images are under
`reference_outputs/full_example/`. They are intentionally versioned because
the paper studies image recognition and the stage images permit visual and
quantitative verification.

The reference output directory is about 447.5 MB. No individual file exceeds
GitHub's 100 MB hard limit, although cloning will be relatively large. For a
long-term journal release, the authors may additionally archive the data with
Zenodo and cite a DOI; the repository should keep either the files or stable
download instructions, never broken placeholders.

## Limitations discovered

- Only one input AVI and one manual reference workbook are publicly packaged.
  The repository also contains all case-study figures supplied for the
  manuscript. The complete research dataset comprises 530 boreholes from the
  Dongzhuang Project. Raw records for the remaining boreholes are not released
  directly because they involve engineering confidentiality; justified requests
  may be directed to the corresponding author and remain subject to data-owner
  authorization.
- Training images and training labels were not supplied; inference is
  reproducible from the included weights, but model training is not.
- The archived output lacks the final whole-borehole evaluation workbook. The
  corrected pipeline generates it on rerun.
- A cold import of the skeleton-analysis stack can trigger one-time Numba
  compilation. The small post-processing smoke test disables JIT only for fast
  validation; the full pipeline keeps JIT enabled.
