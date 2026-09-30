from pathlib import Path
import math
import cv2
import numpy

# Algorithm note: see docs/METHODOLOGY.md.
video_path = str(Path(__file__).resolve().parents[1] / 'data' / 'example' / 'SFGC-BX-007-III-1-9_image.avi')
output_folder = str(Path(__file__).resolve().parents[1] / 'outputs' / 'borehole_segments')
segment_height_m = 5.0
length_m = 46.3
diameter_mm = 76.0
start_depth_m = 0.0


def segment_video(video_path, output_folder, length_m, diameter_mm=76.0,
                  segment_height_m=5.0, start_depth_m=0.0):
    """沿用逐帧翻转拼接方式切图，返回路径、深度、实际长度及像素比例。

    视频必须是原脚本适用的连续展开画幅。length_m 是整个视频覆盖的长度，
    start_depth_m 是视频起始深度。ratio_mm_per_pixel 顺序为纵向、横向。
    导入模块不会打开视频或生成图片。
    """
    for name, value in [('length_m', length_m), ('diameter_mm', diameter_mm),
                        ('segment_height_m', segment_height_m)]:
        if not numpy.isfinite(value) or value <= 0:
            raise ValueError(f'{name} 必须为正数')
    if not numpy.isfinite(start_depth_m):
        raise ValueError('起始深度必须是有限数值')
    capture = cv2.VideoCapture(str(video_path))
    frames = []
    try:
        if not capture.isOpened():
            raise ValueError(f'无法打开视频: {video_path}')
        ok, frame = capture.read()
        while ok:
            flipped = numpy.flip(frame, axis=0)
            ok, frame = capture.read()
            if not ok:
                flipped = flipped[numpy.mean(flipped, axis=(1, 2)) <= 250]
            if flipped.shape[0]:
                frames.append(flipped)
    finally:
        capture.release()
    if not frames:
        raise ValueError('视频没有可用画幅')
    image = numpy.concatenate(frames, axis=0)
    del frames
    perimeter = diameter_mm * numpy.pi
    total_mm = length_m * 1000
    width = int(image.shape[0] / total_mm * perimeter)
    if width < 1:
        raise ValueError('图像像素不足，请检查视频和孔段长度')
    image = cv2.resize(image, (width, image.shape[0]), interpolation=cv2.INTER_AREA)
    quotient = length_m / segment_height_m
    count = round(quotient) if math.isclose(quotient, round(quotient), rel_tol=1e-12) else math.ceil(quotient)
    boundaries = [min(i * segment_height_m, length_m) for i in range(count)] + [length_m]
    rows = [int(depth / length_m * image.shape[0]) for depth in boundaries]
    rows[-1] = image.shape[0]
    if any(b <= a for a, b in zip(rows, rows[1:])):
        raise ValueError('切图高度太小，某段不足一个像素')
    destination = Path(output_folder)
    destination.mkdir(parents=True, exist_ok=True)
    segments = []
    for index, (first, last) in enumerate(zip(rows, rows[1:])):
        start = start_depth_m + boundaries[index]
        end = start_depth_m + boundaries[index + 1]
        path = destination / f'segment_{start:.12g}m_to_{end:.12g}m.png'
        ok, encoded = cv2.imencode('.png', image[first:last])
        if not ok:
            raise RuntimeError(f'图片编码失败: {path}')
        encoded.tofile(str(path))
        segments.append(dict(path=str(path), start_depth_m=start, end_depth_m=end,
                             length_m=boundaries[index + 1] - boundaries[index],
                             diameter_mm=diameter_mm, pixel_start=first, pixel_end=last,
                             ratio_mm_per_pixel=[total_mm / image.shape[0], perimeter / width]))
        print(f'[{index + 1}/{count}] 已保存: {path}')
    return segments


def main():
    return segment_video(video_path, output_folder, length_m, diameter_mm,
                         segment_height_m, start_depth_m)


if __name__ == '__main__':
    main()
