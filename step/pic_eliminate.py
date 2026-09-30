import cv2
import numpy as np
from pathlib import Path

# Algorithm note: see docs/METHODOLOGY.md.

# Algorithm note: see docs/METHODOLOGY.md.
input_folder = str(Path(__file__).resolve().parents[1] / "outputs" / "recognize")
output_folder = str(Path(__file__).resolve().parents[1] / "outputs" / "eliminate")
area_threshold = 100  # See docs/METHODOLOGY.md.
aspect_ratio_threshold = 0.5  # See docs/METHODOLOGY.md.
rectangularity_threshold = 0.5  # See docs/METHODOLOGY.md.

def eliminate_image(image, area_threshold=100, aspect_ratio_threshold=0.5,
                    rectangularity_threshold=0.5):
    """在内存中去噪，保留裂隙原色；返回新的 BGR 图像，不修改输入。"""
    if image is None or image.size == 0 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('需要非空的三通道 BGR 图像')
    for value in (area_threshold, aspect_ratio_threshold, rectangularity_threshold):
        if not np.isfinite(value) or value < 0:
            raise ValueError('筛选阈值必须是非负有限数值')
    img = image
    # Algorithm note: see docs/METHODOLOGY.md.
    binary_img = np.any(img != 0, axis=2).astype(np.uint8)
    
    # Algorithm note: see docs/METHODOLOGY.md.
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    
    # Algorithm note: see docs/METHODOLOGY.md.
    result_img = np.zeros_like(img)
    
    # Algorithm note: see docs/METHODOLOGY.md.
    for i in range(1, num_labels):
        # Algorithm note: see docs/METHODOLOGY.md.
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Algorithm note: see docs/METHODOLOGY.md.
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        
        # Algorithm note: see docs/METHODOLOGY.md.
        if height > 0:  # See docs/METHODOLOGY.md.
            aspect_ratio = width / height
        else:
            aspect_ratio = 0
        
        # Algorithm note: see docs/METHODOLOGY.md.
        if width > 0 and height > 0:  # See docs/METHODOLOGY.md.
            rectangularity = area / (width * height)
        else:
            rectangularity = 0
        
        # Algorithm note: see docs/METHODOLOGY.md.
        # Algorithm note: see docs/METHODOLOGY.md.
        should_keep = (area >= area_threshold) and not (
            aspect_ratio > aspect_ratio_threshold and 
            rectangularity > rectangularity_threshold
        )
        
        if should_keep:
            # Algorithm note: see docs/METHODOLOGY.md.
            component_mask = (labels == i)
            result_img[component_mask] = img[component_mask]
    
    return result_img


def eliminate_small_components(image_path, area_threshold=100, aspect_ratio_threshold=0.5, 
                                rectangularity_threshold=0.5, output_path=None):
    """
    删除图像中满足以下条件的连通域，保留其余区域的原始颜色：
    1. 面积小于指定阈值
    2. 长宽比和矩度均大于指定阈值（长宽比=宽/长，矩度=面积/外接矩形面积）
    
    参数:
    image_path: 输入裂隙识别图路径（黑色背景）
    area_threshold: 面积阈值，小于此值的连通域将被删除
    aspect_ratio_threshold: 长宽比阈值，与矩度阈值同时满足时删除
    rectangularity_threshold: 矩度阈值，与长宽比阈值同时满足时删除
    output_path: 输出图像路径，如果为None则不保存
    
    返回:
    处理后的图像
    """
    # Algorithm note: see docs/METHODOLOGY.md.
    # Algorithm note: see docs/METHODOLOGY.md.
    img = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    
    if img is None:
        print("错误：无法读取图像，请检查文件路径")
        return None
    
    result_img = eliminate_image(img, area_threshold, aspect_ratio_threshold, rectangularity_threshold)

    # Algorithm note: see docs/METHODOLOGY.md.
    if output_path:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        success, encoded = cv2.imencode(destination.suffix, result_img)
        if not success:
            raise RuntimeError(f"结果图片编码失败: {destination}")
        encoded.tofile(str(destination))
    
    return result_img


def main():
    source = Path(input_folder)
    destination = Path(output_folder)
    if not source.is_dir():
        raise NotADirectoryError(f"输入文件夹不存在: {source}")
    if source.resolve() == destination.resolve():
        raise ValueError("输入和输出文件夹必须不同，避免重复处理结果图片")

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
    succeeded = 0
    for index, image_path in enumerate(image_paths, start=1):
        try:
            # Algorithm note: see docs/METHODOLOGY.md.
            output_path = destination / f"{image_path.name}_eliminated.png"
            result = eliminate_small_components(
                image_path,
                area_threshold=area_threshold,
                aspect_ratio_threshold=aspect_ratio_threshold,
                rectangularity_threshold=rectangularity_threshold,
                output_path=output_path,
            )
            if result is None:
                raise ValueError("无法读取图片")
            succeeded += 1
            print(f"[{index}/{len(image_paths)}] 已保存: {output_path}")
        except Exception as exc:
            print(f"[{index}/{len(image_paths)}] 处理失败: {image_path.name}: {exc}")
    print(f"处理完成：成功 {succeeded} 张，失败 {len(image_paths) - succeeded} 张。保存位置: {destination}")


if __name__ == "__main__":
    main()
