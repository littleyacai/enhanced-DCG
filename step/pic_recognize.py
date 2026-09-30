from pathlib import Path
import cv2
import numpy

# Standalone defaults. The main pipeline passes paths explicitly.
input_folder = str(Path(__file__).resolve().parents[1] / "outputs" / "borehole_segments")
output_folder = str(Path(__file__).resolve().parents[1] / "outputs" / "recognize")
segment_length = 1000


def create_recognizer(model_path=None, device="auto"):
    """Load the segmentation model once and reuse it for all image segments."""
    import importlib
    module = importlib.import_module('step.lib.image.deeplab')
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    use_cuda = device == "cuda" or (device == "auto" and module.torch.cuda.is_available())
    if device == "cuda" and not module.torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    kwargs = {"cuda": use_cuda}
    if model_path is not None:
        kwargs["model_path"] = str(Path(model_path).expanduser().resolve())
    model = module.DeeplabV3(**kwargs)
    # Class 3 is cemented fracture; use yellow in the BGR output image.
    model.colors[3] = (0, 128, 128)
    return model


def recognize(image, deeplab, segment_length=1000):
    """Recognize one image in vertical tiles and restore its original size."""
    if not isinstance(segment_length, int) or segment_length <= 0:
        raise ValueError('识别高度必须是正整数（像素）')
    if image is None or image.size == 0:
        raise ValueError('输入图片为空')
    height, width = image.shape[:2]
    final_result = numpy.zeros((height, width, 3), dtype=numpy.uint8)
    for start in range(0, height, segment_length):
        end = min(start + segment_length, height)
        result = deeplab.detect_image(image[start:end, :])
        if result.shape[:2] != (end - start, width):
            result = cv2.resize(result, (width, end - start), interpolation=cv2.INTER_NEAREST)
        final_result[start:end, :] = result
    return final_result


def recognize_file(image_path, output_path, deeplab, segment_length=1000):
    """识别一张孔段图片并保存，返回彩色结果数组。"""
    image = cv2.imdecode(numpy.fromfile(str(image_path), dtype=numpy.uint8), cv2.IMREAD_COLOR)
    result = recognize(image, deeplab, segment_length)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(destination.suffix, result)
    if not ok:
        raise RuntimeError(f'图片编码失败: {destination}')
    encoded.tofile(str(destination))
    return result


def main():
    if not isinstance(segment_length, int) or segment_length <= 0:
        raise ValueError("识别高度必须是大于 0 的整数（像素）")
    source = Path(input_folder)
    destination = Path(output_folder)
    if not source.is_dir():
        raise NotADirectoryError(f"输入文件夹不存在: {source}")
    if source.resolve() == destination.resolve():
        raise ValueError("输入和输出文件夹必须不同，避免重复识别结果图片")

    # Algorithm note: see docs/METHODOLOGY.md.
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    image_paths = sorted(
        path for path in source.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )
    if not image_paths:
        print(f"输入文件夹中没有支持的图片: {source}")
        return
    destination.mkdir(parents=True, exist_ok=True)

    deeplab = create_recognizer()

    succeeded = 0
    for index, image_path in enumerate(image_paths, start=1):
        try:
            # Algorithm note: see docs/METHODOLOGY.md.
            image = cv2.imdecode(numpy.fromfile(str(image_path), dtype=numpy.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("无法读取图片")
            final_result = recognize(image, deeplab, segment_length)
            # Algorithm note: see docs/METHODOLOGY.md.
            output_path = destination / f"{image_path.name}_recognition_result.png"
            encoded_ok, encoded = cv2.imencode(".png", final_result)
            if not encoded_ok:
                raise RuntimeError("结果图片编码失败")
            encoded.tofile(str(output_path))
            succeeded += 1
            print(f"[{index}/{len(image_paths)}] 已保存: {output_path}")
        except Exception as exc:
            print(f"[{index}/{len(image_paths)}] 处理失败: {image_path.name}: {exc}")
    print(f"处理完成：成功 {succeeded} 张，失败 {len(image_paths) - succeeded} 张。保存位置: {destination}")


if __name__ == "__main__":
    main()
