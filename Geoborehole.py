
from pathlib import Path
import argparse
import math


# Default paths and parameters for the public example dataset.
BASE_DIR = Path(__file__).resolve().parent
VIDEO_PATH = BASE_DIR / 'data' / 'example' / 'SFGC-BX-007-III-1-9_image.avi'
OUTPUT_ROOT = BASE_DIR / 'outputs' / 'example_run'
STANDARD_EXCEL_PATH = BASE_DIR / 'data' / 'example' / 'SFGC-BX-007-III-1-9.xlsx'
MODEL_PATH = BASE_DIR / 'step' / 'lib' / 'image' / 'logs' / 'ep300-loss0.031-val_loss0.053.pth'

BOREHOLE_LENGTH_M = 46.3
BOREHOLE_DIAMETER_MM = 76.0
SEGMENT_HEIGHT_M = 5.0

AREA_THRESHOLD = 100
ASPECT_RATIO_THRESHOLD = 0.5
RECTANGULARITY_THRESHOLD = 0.5
REFERENCE_POINT_IDX = 50
NUM_EXPAND_POINTS = 10


def run_pipeline(video_path, output_root, standard_excel_path, *, length_m=45.9,
                 start_depth_m=0.0, diameter_mm=76.0, segment_height_m=5.0,
                 recognition_height_px=1000, area_threshold=100,
                 aspect_ratio_threshold=0.5, rectangularity_threshold=0.5,
                 reference_point_idx=50, num_expand_points=10, pruning_times=1,
                 branch_length_threshold=20, clustering_mean_threshold=1.7,
                 clustering_std_threshold=1.5, standard_sheet=0,
                 model_path=MODEL_PATH, device="auto"):
    """Run the complete workflow for one borehole video.

    The output directory must be empty so that results from different runs
    cannot be mixed accidentally.
    """
    from step.pic_segment import segment_video
    from step.pic_recognize import create_recognizer, recognize_file
    from step.pic_eliminate import eliminate_image
    from step.pic_extension import extend_image
    from step.pic_clustering import cluster_segment, save_image
    from step.evaluation import load_standard_excel, save_borehole_report

    video_path = Path(video_path).resolve()
    standard_excel_path = Path(standard_excel_path).resolve()
    root = Path(output_root).resolve()
    for path in (video_path, standard_excel_path):
        if not path.is_file():
            raise FileNotFoundError(f'Input file does not exist: {path}')
    for name, value in [('孔长', length_m), ('钻孔直径', diameter_mm), ('切段长度', segment_height_m),
                        ('骨架长度阈值', branch_length_threshold),
                        ('聚类均值阈值', clustering_mean_threshold), ('聚类标准差阈值', clustering_std_threshold)]:
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be positive')
    if not math.isfinite(start_depth_m):
        raise ValueError('start_depth_m must be finite')
    for name, value, minimum in [('识别高度', recognition_height_px, 1),
                                  ('参考点索引', reference_point_idx, 1),
                                  ('扩展点数', num_expand_points, 0), ('修剪次数', pruning_times, 0)]:
        if not isinstance(value, int) or value < minimum:
            raise ValueError(f'{name} must be an integer greater than or equal to {minimum}')
    for value in (area_threshold, aspect_ratio_threshold, rectangularity_threshold):
        if not math.isfinite(value) or value < 0:
            raise ValueError('Filtering thresholds must be finite and non-negative')
    standard = load_standard_excel(standard_excel_path, standard_sheet)
    if not standard['深度(m)'].between(start_depth_m, start_depth_m + length_m).all():
        raise ValueError('Reference depths fall outside the configured borehole interval')
    folders = {name: root/name for name in ('borehole_segments', 'recognize', 'eliminate', 'extension', 'result')}
    for folder in folders.values():
        if folder.exists() and (not folder.is_dir() or any(folder.iterdir())):
            raise FileExistsError(f'Output folder is not empty; choose another output root: {folder}')
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)

    print('Loading the four-class segmentation model', flush=True)
    model = create_recognizer(model_path=model_path, device=device)
    segments = segment_video(video_path, folders['borehole_segments'], length_m,
                             diameter_mm, segment_height_m, start_depth_m)
    results = []
    for index, segment in enumerate(segments, start=1):
        name = Path(segment['path']).name
        print(f'\nSegment {index}/{len(segments)}: {name}', flush=True)
        paths = {stage: folders[stage]/name for stage in ('recognize', 'eliminate', 'extension')}
        recognition = recognize_file(segment['path'], paths['recognize'], model, recognition_height_px)
        eliminated = eliminate_image(recognition, area_threshold, aspect_ratio_threshold, rectangularity_threshold)
        save_image(paths['eliminate'], eliminated)
        del recognition
        extended, endpoints = extend_image(eliminated, reference_point_idx, num_expand_points)
        save_image(paths['extension'], extended)
        del eliminated, extended, endpoints
        result = cluster_segment(paths['extension'], segment, folders['result'],
            pruning_times=pruning_times, branch_length_threshold=branch_length_threshold,
            clustering_mean_threshold=clustering_mean_threshold, clustering_std_threshold=clustering_std_threshold,
            save_intermediate_images=False)
        # Retain only parameters and segment metadata for the borehole report.
        results.append(dict(fracture=result['fracture'], segment=result['segment']))
        del result
    del model
    report = save_borehole_report(results, folders['result'], standard_excel_path, length_m,
                                 start_depth_m, filter_to_segment=False, sheet_name=standard_sheet)
    print(report['metrics'].to_string(index=False), flush=True)
    print(f'Processing complete. Report: {report["path"]}', flush=True)
    return dict(folders={k: str(v) for k, v in folders.items()}, report=report)


def build_parser():
    parser = argparse.ArgumentParser(description="Filling-aware fracture reconstruction from panoramic borehole imagery")
    parser.add_argument("--video", type=Path, default=VIDEO_PATH, help="Input panoramic borehole AVI")
    parser.add_argument("--reference", type=Path, default=STANDARD_EXCEL_PATH, help="Manual reference XLSX")
    parser.add_argument("--weights", type=Path, default=MODEL_PATH, help="DeepLabV3+ model weights")
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT, help="Empty output directory")
    parser.add_argument("--length-m", type=float, default=BOREHOLE_LENGTH_M)
    parser.add_argument("--start-depth-m", type=float, default=0.0)
    parser.add_argument("--diameter-mm", type=float, default=BOREHOLE_DIAMETER_MM)
    parser.add_argument("--segment-height-m", type=float, default=SEGMENT_HEIGHT_M)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return run_pipeline(
        args.video, args.output, args.reference,
        length_m=args.length_m, start_depth_m=args.start_depth_m,
        diameter_mm=args.diameter_mm, segment_height_m=args.segment_height_m,
        area_threshold=AREA_THRESHOLD, aspect_ratio_threshold=ASPECT_RATIO_THRESHOLD,
        rectangularity_threshold=RECTANGULARITY_THRESHOLD,
        reference_point_idx=REFERENCE_POINT_IDX, num_expand_points=NUM_EXPAND_POINTS,
        model_path=args.weights, device=args.device,
    )


if __name__ == '__main__':
    main()
