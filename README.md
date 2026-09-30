# Enhanced DCG

Reproducible code and example data for an image-based workflow that recognizes,
reconnects, reconstructs, and evaluates fractures in panoramic borehole images.
The repository accompanies the manuscript submitted to *Computers &
Geosciences*.

## What is included

- The complete processing code, including the trained DeepLabV3+ weights.
- All case-study figures and associated materials presented directly in the
  manuscript.
- One 46.3 m panoramic borehole video and its manual fracture log.
- All PNG intermediate and final outputs for that public example.
- The fracture-parameter spreadsheets generated for the ten depth intervals.
- Small, fast tests and repository-validation scripts.

The complete Dongzhuang Project dataset used in the study comprises 530
boreholes. The repository contains the case-study materials presented in the
manuscript and one complete public example that exercises every stage of the
published workflow. The remaining project records are subject to engineering
confidentiality requirements and are not publicly released. Consequently, the
public example cannot by itself reproduce aggregate statistics calculated over
the complete 530-borehole dataset. Researchers with a justified need may
contact the corresponding author to discuss access; any release remains subject
to confidentiality review and authorization by the data owner.

## Repository layout

```text
Geoborehole.py                   Command-line entry point
step/                            Processing and evaluation modules
  lib/image/logs/*.pth           Trained segmentation weights
data/example/                    Input AVI and manual XLSX reference
data/case_study_figures/         Figures used for qualitative case discussion
reference_outputs/full_example/  All PNG/XLSX outputs from the archived run
reference_outputs/segment_tables Archived per-segment parameter tables
docs/                            Method, data, and reproduction notes
scripts/                         Automated checks and reference-data summaries
tests/                           Fast unit and smoke tests
```

## Installation

Python 3.10 is recommended. Create a clean environment and install the tested
dependencies:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch wheels are platform-specific. If the command above cannot provide the
appropriate CPU or CUDA build, install PyTorch first using the official PyTorch
selector, then install the remaining requirements.

The algorithm uses the NVIDIA CUDA parallel-computing platform through PyTorch
to accelerate DeepLabV3+ inference and other tensor operations. A compatible
NVIDIA GPU, driver, CUDA-enabled PyTorch build, and CUDA runtime are recommended
for reproducing the full borehole analysis efficiently. CPU execution is also
supported for portability, but it is substantially slower and was not the
primary computational configuration used for the study.

## Quick checks

Run these before a full reproduction:

```bash
python scripts/check_repository.py
python scripts/validate_reference_outputs.py
python -m unittest discover -s tests -v
python scripts/model_smoke_test.py
python scripts/postprocess_smoke_test.py
```

The model test loads the weights on CPU and processes a small image. The
post-processing test checks filtering, reconnection, skeleton clustering, and
fracture fitting on a deterministic synthetic mask. Neither requires a GPU.

## Reproduce the example

From the repository root:

```bash
python Geoborehole.py --device auto --output outputs/example_run
```

Use `--device cpu` to force CPU execution or `--device cuda` to require a CUDA
GPU. The defaults point only to files inside this repository; no author-specific
absolute paths are used. To process another borehole:

```bash
python Geoborehole.py \
  --video path/to/borehole.avi \
  --reference path/to/manual_log.xlsx \
  --length-m 46.3 \
  --diameter-mm 76 \
  --segment-height-m 5 \
  --output outputs/my_run
```

The output directory must be new or empty. A complete run produces five image
stage folders plus `result/borehole_evaluation.xlsx`. See
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the exact archived
parameters and expected file inventory.

## Image classes and colors

The segmentation output uses four pixel classes:

| Class | Meaning | Stored BGR value | Display color |
|---|---|---:|---|
| 0 | background | `(0, 0, 0)` | black |
| 1 | filled fracture | `(0, 0, 128)` | dark red |
| 2 | unfilled fracture | `(0, 128, 0)` | dark green |
| 3 | cemented fracture | `(0, 128, 128)` | dark yellow |

Mixed reconnections may be represented in blue during post-processing. Details
and coordinate conventions are in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## Reproducibility status

The public package was checked for Python syntax, missing repository files,
hard-coded Windows drive paths, file-size limits, workbook readability, model
loading, and key numerical routines. Reference PNGs are intentionally retained:
they are scientific intermediate data, not decorative assets. The much smaller
recognition/elimination/extension images are class masks; the large segment and
2-D result images preserve the original image information.

The complete source-folder inventory and the rationale for retained/excluded
files are recorded in [`docs/SOURCE_AUDIT.md`](docs/SOURCE_AUDIT.md).

## Citation

Please cite the accompanying manuscript. Repository metadata are also provided
in [`CITATION.cff`](CITATION.cff); update the DOI and publication fields after
acceptance.

## Licenses

Source code is released under the [MIT License](LICENSE). The example data,
reference outputs, case-study figures, and model weights actually included in
this repository are released under [CC BY 4.0](LICENSE-DATA.md). This license
does not apply to the confidential 530-borehole project dataset, which is not
distributed through this repository.

## Contact

Tuo Li (corresponding author), School of Civil and Transportation Engineering,
Hebei University of Technology. Email: 2021095@hebut.edu.cn. Requests for the
confidential 530-borehole Dongzhuang Project dataset should explain the intended
research use and remain subject to the necessary project authorization.
