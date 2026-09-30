# Methodology and implementation map

## Processing sequence

1. **Panoramic image assembly and depth segmentation** (`step/pic_segment.py`)
   flips and concatenates the AVI frames, rescales the circumference to
   `diameter × π`, and divides the resulting image by physical depth.
2. **Four-class semantic segmentation** (`step/pic_recognize.py`) applies a
   DeepLabV3+ network with a MobileNetV2 backbone. Tall interval images are
   processed in 1,000-pixel vertical tiles.
3. **False-positive filtering** (`step/pic_eliminate.py`) removes connected
   components below the area threshold and compact components that jointly
   exceed the aspect-ratio and rectangularity criteria.
4. **Trace reconnection** (`step/pic_extension.py`) skeletonizes fracture
   masks, estimates terminal directions and local width, expands compatible
   endpoints, and assigns the reconnected region according to the connected
   fracture classes.
5. **Skeleton pruning and clustering** (`step/pic_clustering.py`) separates
   branches, associates compatible branches, transforms pixels to a local
   three-dimensional borehole coordinate system, fits fracture planes, and
   derives depth, aperture, trace length, dip direction, and dip angle.
6. **Whole-borehole evaluation** (`step/evaluation.py`) compares predictions
   with the manual log and exports a multi-sheet XLSX report.

## Coordinate and angle conventions

- Depth is measured in metres along the borehole.
- Aperture and trace length are reported in millimetres.
- Dip direction is an azimuth in degrees clockwise from north, in `[0, 360)`.
- Dip angle is in degrees from horizontal, in `[0, 90]`.
- The local fitting frame is right-handed: north, east, and vertical/depth.
- A horizontal fracture has no unique dip direction; evaluation assigns an
  arbitrary azimuth of zero because its unit normal is unchanged.

## Archived example parameters

| Parameter | Value |
|---|---:|
| Borehole length | 46.3 m |
| Borehole diameter | 76 mm |
| Depth interval | 5 m (last interval 1.3 m) |
| Recognition tile height | 1,000 pixels |
| Component area threshold | 100 pixels |
| Aspect-ratio threshold | 0.5 |
| Rectangularity threshold | 0.5 |
| Endpoint reference index | 50 skeleton points |
| Maximum expansion steps | 10 |
| Skeleton pruning passes | 1 |
| Minimum branch length | 20 |
| Clustering mean threshold | 1.7 |
| Clustering standard-deviation threshold | 1.5 |

These values are explicit defaults in `Geoborehole.py`; command-line options
control the borehole geometry, files, output location, and compute device.

## Evaluation implementation

For a reference fracture `i` and predicted fracture `j`, the implementation
computes a position similarity using the absolute depth difference divided by
the full borehole length, an angular similarity using the absolute dot product
of unit normal vectors, and their product. A Hungarian assignment is solved
separately for each score matrix. Unmatched fractures contribute zero through
zero padding, and the total is divided by the larger fracture count.

These labels are retained for continuity with the archived code (`QPI`, `QAI`,
and `CEI`). They are implementation-specific evaluation quantities, not
community-standard metrics. The public repository therefore documents the
equations rather than relying on the abbreviations alone.

## Important implementation choices

- The pipeline keeps all fitted fractures during whole-borehole evaluation by
  default (`filter_to_segment=False`), matching the archived workflow.
- The result directory must be empty to prevent accidental mixing of runs.
- Input files and model weights are located relative to the repository root.
- Model inference automatically falls back to CPU if CUDA is unavailable.
- XLSX files are written with `openpyxl`; no proprietary or assistant-specific
  runtime is required.
