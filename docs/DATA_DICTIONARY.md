# Data dictionary

## `data/example/SFGC-BX-007-III-1-9_image.avi`

Panoramic borehole source video for the public example. The archived file has
154 frames at 25 frames/s and a frame size of 2,048 × 1,200 pixels. The video
represents a 46.3 m interval starting at 0 m in a 76 mm diameter borehole.

## `data/example/SFGC-BX-007-III-1-9.xlsx`

Manual fracture log. The first row is a merged title and the next row contains
the usable field names.

| Original field | Meaning | Unit/encoding |
|---|---|---|
| 序号 | record number | integer |
| 深度 [m] | fracture depth | m |
| 倾向 [deg] | dip direction | degrees clockwise from north |
| 倾角 [deg] | dip angle | degrees from horizontal |
| 开度 [mm] | aperture | mm |
| 类型 | fracture type | source categorical label |

Blank reserved rows are ignored. The evaluator locates the depth, dip-direction,
and dip-angle columns by their labels rather than by fixed positions.

## `reference_outputs/full_example/`

The complete archived PNG/XLSX result set, retained because images are primary
scientific data in this study.

| Folder | Contents |
|---|---|
| `borehole_segments/` | depth-calibrated color interval images |
| `recognize/` | four-class model masks |
| `eliminate/` | masks after connected-component filtering |
| `extension/` | masks after endpoint reconnection |
| `result/` | reconstructed 2-D images and per-segment XLSX tables |

Large color interval and final images retain the original texture and are
therefore much larger than the class-mask PNGs. They are not duplicates of the
mask images.

## Segment parameter tables

The primary fields exported by `fracture_parameter_table` are:

| Field | Meaning |
|---|---|
| 裂隙编号 | fracture identifier within the exported table |
| 裂隙类型 | human-readable fracture class |
| 类型标识 | machine-readable class (`filled`, `unfilled`, `cemented`, `halffilled`) |
| 深度(m) | fitted fracture-centre depth |
| 开度(mm) | mean estimated aperture |
| 迹线长度(mm) | reconstructed trace length |
| 开度乘迹线长度(mm²) | aperture–trace product |
| 倾向(°) | dip direction |
| 倾角(°) | dip angle |
| 法向量X/Y/Z | fitted plane unit-normal components |
| 骨架点数 | number of skeleton pixels in the fitted cluster |

## Case-study figures

`data/case_study_figures/` contains seven publication-oriented qualitative
figures. These are the complete case-study figure files provided for the
manuscript. They are supporting data and are not required by the command-line
pipeline.

## Public-data scope

The repository packages all case-study materials presented directly in the
manuscript and one complete example borehole for end-to-end reproduction. The
complete Dongzhuang Project dataset comprises 530 boreholes. Raw records for
the remaining boreholes involve project confidentiality and therefore are not
released publicly. Researchers with a justified need may contact the
corresponding author, Tuo Li (2021095@hebut.edu.cn), to discuss a data request.
Access, if possible, remains subject to confidentiality review, the intended
use, and authorization by the data owner; contacting the author does not
guarantee that access can be granted.
