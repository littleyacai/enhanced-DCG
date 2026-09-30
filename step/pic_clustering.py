from pathlib import Path
import sys
import cv2
import numpy
import pandas
from skan import csr
from skimage import morphology
from skspatial.objects import Plane, Points

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from .lib.fracture.abstract import getAperture, getTrace
except ImportError:
    from lib.fracture.abstract import getAperture, getTrace

# Algorithm note: see docs/METHODOLOGY.md.
recognition_image_path = str(Path(__file__).resolve().parents[1] / 'tests' / 'fixtures' / 'recognition_crop.png')
# Algorithm note: see docs/METHODOLOGY.md.
original_image_path = r""
output_folder = str(Path(__file__).resolve().parents[1] / 'outputs' / 'standalone_clustering')
segment_length_m = 5.0
start_depth_m = 15.0  # See docs/METHODOLOGY.md.
borehole_diameter_mm = 76.0
pruning_times = 1
branch_length_threshold = 20
clustering_mean_threshold = 1.7
clustering_std_threshold = 1.5
save_intermediate_images = True
GLOBAL = 'upright'


def read_image(path):
    image = cv2.imdecode(numpy.fromfile(str(path), dtype=numpy.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片: {path}")
    return image


def save_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode('.png', image)
    if not ok:
        raise RuntimeError(f"无法编码图片: {path}")
    data.tofile(str(path))


def outline(img, thresh_area=100):
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if cv2.contourArea(c) > thresh_area]
    return [[tuple(p[0]) for p in c] for c in contours], [cv2.contourArea(c) for c in contours]


def has_edges(skeleton):
    neighbors = cv2.filter2D((skeleton > 0).astype(numpy.uint8), -1,
                             numpy.ones((3, 3), numpy.uint8), borderType=cv2.BORDER_CONSTANT)
    return bool(numpy.any((skeleton > 0) & (neighbors > 1)))


def getparameter(image, debug=False, *, length_m=5.0, start_depth_m=0.0,
                 diameter_mm=76.0, ratio_mm_per_pixel=None):
    """使用当前孔段参数；比例顺序为纵向、横向（毫米/像素）。"""
    if image is None or image.size == 0:
        raise ValueError('输入图片为空')
    if not numpy.isfinite(length_m) or length_m <= 0:
        raise ValueError('孔段长度必须为正数')
    if not numpy.isfinite(diameter_mm) or diameter_mm <= 0:
        raise ValueError('钻孔直径必须为正数')
    if not numpy.isfinite(start_depth_m):
        raise ValueError('起始深度必须为有限数值')
    perimeter = diameter_mm * numpy.pi
    # Algorithm note: see docs/METHODOLOGY.md.
    ratio = numpy.asarray(ratio_mm_per_pixel if ratio_mm_per_pixel is not None
                          else [length_m * 1000 / image.shape[0], perimeter / image.shape[1]], dtype=float)
    if ratio.shape != (2,) or not numpy.all(numpy.isfinite(ratio)) or numpy.any(ratio <= 0):
        raise ValueError('像素比例必须是两个正数：纵向、横向')
    return dict(length=length_m, start=start_depth_m, end=start_depth_m + length_m,
                diameter=diameter_mm, radius=diameter_mm/2, perimeter=perimeter,
                ratio=ratio, path='./', name='result')



def fracget(recognition_result):
    """
    从彩色识别图片中提取不同颜色的裂隙掩膜并生成二值化图片
    
    参数:
        recognition_result: 彩色识别图片 (BGR格式)
    
    返回:
        masks: 字典，包含各种裂隙类型的掩膜
        binary_image: 二值化图片 (背景黑色，裂隙白色)
    """
    # Algorithm note: see docs/METHODOLOGY.md.
    color_ranges = {
        'filled': (0, 0, 128),  # See docs/METHODOLOGY.md.
        'unfilled': (0, 128, 0),  # See docs/METHODOLOGY.md.
        'cemented': (0, 128, 128),  # See docs/METHODOLOGY.md.
        'halffilled': (128, 0, 0)  # See docs/METHODOLOGY.md.
    }
    
    # Algorithm note: see docs/METHODOLOGY.md.
    masks = {}
    
    # Algorithm note: see docs/METHODOLOGY.md.
    for fracture_type, target_color in color_ranges.items():
        # Algorithm note: see docs/METHODOLOGY.md.
        lower_bound = numpy.array([max(0, c-10) for c in target_color])
        upper_bound = numpy.array([min(255, c+10) for c in target_color])
        
        mask = cv2.inRange(recognition_result, lower_bound, upper_bound)
        # Algorithm note: see docs/METHODOLOGY.md.
        bright = numpy.array([255 if c else 0 for c in target_color])
        bright_lower = numpy.clip(bright - 10, 0, 255).astype(numpy.uint8)
        bright_upper = numpy.clip(bright + 10, 0, 255).astype(numpy.uint8)
        mask |= cv2.inRange(recognition_result, bright_lower, bright_upper)
        masks[fracture_type] = mask
    
    # Algorithm note: see docs/METHODOLOGY.md.
    binary_image = numpy.zeros(recognition_result.shape[:2], dtype=numpy.uint8)
    
    # Algorithm note: see docs/METHODOLOGY.md.
    for mask in masks.values():
        binary_image = cv2.bitwise_or(binary_image, mask)

    return masks, binary_image



def branchGeneration(binary_image, times=1, thresLength=20, debug=False):
    # Algorithm note: see docs/METHODOLOGY.md.
    skeleton = morphology.skeletonize(binary_image)
    # Algorithm note: see docs/METHODOLOGY.md.
    for iter in range(times):
        # Algorithm note: see docs/METHODOLOGY.md.
        if not has_edges(skeleton):
            return skeleton.astype(numpy.uint8) * 255
        branches = csr.Skeleton(skeleton)
        info = csr.summarize(branches, separator='-')
        # Algorithm note: see docs/METHODOLOGY.md.
        allPoints = list(info['node-id-src']) + list(info['node-id-dst'])
        intPoints = list(set([n for n in allPoints if allPoints.count(n) > 1]))
        # Algorithm note: see docs/METHODOLOGY.md.
        for i in info.loc[(info['branch-type'] == 1)].index:
            if branches.paths[i].size < thresLength:
                branchPath = branches.paths[i].indices[1:] if info['node-id-src'][i] in intPoints else branches.paths[i].indices[:-1]
                for j in branchPath:
                    skeleton[branches.coordinates[j][0], branches.coordinates[j][1]] = False
    # Algorithm note: see docs/METHODOLOGY.md.
    if not has_edges(skeleton):
        return skeleton.astype(numpy.uint8) * 255
    branches = csr.Skeleton(skeleton)
    info = csr.summarize(branches, separator='-')
    for i in range(branches.n_paths):
        if (info['branch-type'][i] == 0 and branches.paths[i].size < thresLength) or (info['branch-type'][i] == 3) or \
           (info['branch-type'][i] == 2 and branches.paths[i].size < thresLength/5):
            for j in branches.paths[i].indices:
                skeleton[branches.coordinates[j][0], branches.coordinates[j][1]] = False
    # Algorithm note: see docs/METHODOLOGY.md.
    branches = skeleton.astype(numpy.uint8)*255
    return branches



def branchClustering(branches, thresMean=1.5, thresVar=1.5, debug=False, masks=None):
    if not has_edges(branches):
        return []
    # Algorithm note: see docs/METHODOLOGY.md.
    perimeter = branches.shape[1]
    radius = perimeter/(2*numpy.pi)
    branches = csr.Skeleton(branches.astype(bool))
    info = csr.summarize(branches, separator='-')
    
    # Algorithm note: see docs/METHODOLOGY.md.
    branches = [[len(branches.path_coordinates(i)), branches.path_coordinates(i)] for i in range(branches.n_paths)]
    branches = [[numpy.mean(i[1], axis=0)[0]] + i for i in sorted(branches, key=lambda x: x[0], reverse=True)]
    branches = pandas.DataFrame(branches, columns=['center', 'length', 'coordinates'])
    # Algorithm note: see docs/METHODOLOGY.md.
    if masks is not None and isinstance(masks, dict) and len(masks) > 0:
        # Algorithm note: see docs/METHODOLOGY.md.
        any_mask = next(iter(masks.values()))
        img_h, img_w = any_mask.shape[:2]

        def classify_fractype(coords):
            if coords is None or len(coords) == 0:
                return 'unknown'
            coord_arr = numpy.asarray(coords)
            coord_idx = numpy.round(coord_arr).astype(int)
            # Algorithm note: see docs/METHODOLOGY.md.
            valid = (coord_idx[:, 0] >= 0) & (coord_idx[:, 0] < img_h) & (coord_idx[:, 1] >= 0) & (coord_idx[:, 1] < img_w)
            coord_idx = coord_idx[valid]
            if coord_idx.size == 0:
                return 'unknown'
            # Algorithm note: see docs/METHODOLOGY.md.
            max_hits = 0
            best_type = 'unknown'
            for frac_type, mask in masks.items():
                hits = int(numpy.count_nonzero(mask[coord_idx[:, 0], coord_idx[:, 1]]))
                if hits > max_hits:
                    max_hits = hits
                    best_type = frac_type
            return best_type if max_hits > 0 else 'unknown'

        branches['fractype'] = branches['coordinates'].apply(classify_fractype)
    else:
        branches['fractype'] = 'unknown'
    # Algorithm note: see docs/METHODOLOGY.md.
    def compute_k(type_a: str, type_b: str) -> float:
        a = (type_a or 'unknown')
        b = (type_b or 'unknown')
        # Algorithm note: see docs/METHODOLOGY.md.
        if a == b and a != 'unknown':
            return 1.2
        # Algorithm note: see docs/METHODOLOGY.md.
        if (a == 'cemented' and b != 'cemented') or (b == 'cemented' and a != 'cemented'):
            return 0.6

        pair = {a, b}
        # Algorithm note: see docs/METHODOLOGY.md.
        if pair == {'filled', 'halffilled'}:
            return 1.0
        # Algorithm note: see docs/METHODOLOGY.md.
        if pair == {'filled', 'unfilled'}:
            return 0.8
        # Algorithm note: see docs/METHODOLOGY.md.
        if pair == {'unfilled', 'halffilled'}:
            return 1.0
        # Algorithm note: see docs/METHODOLOGY.md.
        return 1.0
    # Algorithm note: see docs/METHODOLOGY.md.
    clusterSet = []
    while len(branches) > 1:
        # Algorithm note: see docs/METHODOLOGY.md.
        if branches.iloc[0]['length'] < perimeter/4:
            break
        # Algorithm note: see docs/METHODOLOGY.md.
        coordinates, center, fractype = branches.iloc[0]['coordinates'], branches.iloc[0]['center'], branches.iloc[0]['fractype']
        branches = branches.drop(index=branches.index[[0]])
        # Algorithm note: see docs/METHODOLOGY.md.
        zmin, zmax = center-perimeter*3, center+perimeter*3
        index = [i for i in branches[(branches['center'] > zmin) & (branches['center'] < zmax)].index]
        # Algorithm note: see docs/METHODOLOGY.md.
        while True:
            # Algorithm note: see docs/METHODOLOGY.md.
            coord3D = [[radius*numpy.cos(2*numpy.pi*x/perimeter), radius*numpy.sin(2*numpy.pi*x/perimeter), z] for z, x in coordinates]
            plane = Plane.best_fit(Points(coord3D), full_matrices=False)
            dist = (numpy.asarray(coord3D) - numpy.asarray(plane.point)) @ numpy.asarray(plane.normal)
            mean0, var0, indicate = numpy.mean(abs(dist)), numpy.std(abs(dist)), len(index)
            for i in index:
                coord2D = numpy.concatenate([coordinates, branches['coordinates'][i]], axis=0)
                coord3D = [[radius*numpy.cos(2*numpy.pi*x/perimeter), radius*numpy.sin(2*numpy.pi*x/perimeter), z] for z, x in coord2D]
                plane = Plane.best_fit(Points(coord3D), full_matrices=False)
                dist = (numpy.asarray(coord3D) - numpy.asarray(plane.point)) @ numpy.asarray(plane.normal)
                mean1, var1, expand = numpy.mean(abs(dist)), numpy.std(abs(dist)), len(coord2D)/len(coordinates)
                # Algorithm note: see docs/METHODOLOGY.md.
                k = compute_k(fractype, branches['fractype'][i])
                if mean1 < min([k*thresMean*expand*mean0, perimeter/50]) and var1 < min([k*thresVar*expand*var0, perimeter/50]):
                    coordinates = coord2D
                    branches = branches.drop(i)
                    index.pop(index.index(i))
                    break
            if len(index) == indicate:
                break
        # Algorithm note: see docs/METHODOLOGY.md.
        print(f"Completed cluster {len(clusterSet) + 1}; {len(branches)} branches remain", flush=True)
        clusterSet.append({'coordinates': coordinates, 'fractype': fractype})
    return clusterSet



def cracksGeneration(recognize, cluster, info):
    # Algorithm note: see docs/METHODOLOGY.md.
    ratio, start, end = info['ratio'], info['start'], info['end']
    radius, perimeter, length = info['diameter'] / 2, info['diameter'] * numpy.pi, 1000 * info['length']
    # Algorithm note: see docs/METHODOLOGY.md.
    data = []
    contours, areas = outline(recognize)
    for i in range(len(cluster)):
        print(f"Calculating fracture parameters {i + 1}/{len(cluster)}", flush=True)
        # Algorithm note: see docs/METHODOLOGY.md.
        if isinstance(cluster[i], dict):
            coords_i = cluster[i]['coordinates']
            frac_type = cluster[i].get('fractype', 'unknown')
        else:
            coords_i = cluster[i]
            frac_type = 'unknown'
        # Algorithm note: see docs/METHODOLOGY.md.
        coord2D = coords_i * ratio
        coord3D = [
            [radius * numpy.cos(2 * numpy.pi * x / perimeter), radius * numpy.sin(2 * numpy.pi * x / perimeter), z] for
            z, x in coord2D]
        plane = Plane.best_fit(Points(coord3D), full_matrices=False)
        # Algorithm note: see docs/METHODOLOGY.md.
        if abs(plane.normal[2]) < 0.10:
            continue
        center = sum(plane.normal * plane.point) / plane.normal[2]
        if (center < -3 * radius) or (center > length + 3 * radius):
            continue
        if abs(plane.normal[2]) < 0.10:
            continue
        # Algorithm note: see docs/METHODOLOGY.md.
        center = start + center / 1000
        x, y, z = plane.normal
        normal = numpy.array([y, x, -z]) if GLOBAL == 'upright' else numpy.array([x, y, z])
        normal = normal if normal[2] > 0 else -normal
        angle = abs(numpy.arccos(normal[2]))
        norm_xy = numpy.linalg.norm(normal[:2])
        if norm_xy < 1e-12:
            dip = numpy.nan  # See docs/METHODOLOGY.md.
        else:
            nxy = normal[:2] / norm_xy
            north = numpy.clip(nxy[1], -1, 1)
            dip = numpy.arccos(north) if nxy[0] > 0 else 2 * numpy.pi - numpy.arccos(north)
        # Algorithm note: see docs/METHODOLOGY.md.
        aperture = getAperture(coords_i, contours, info['ratio'])
        trace = getTrace(coords_i, info['ratio'])
        flaw = aperture * trace
        data.append([center, normal, aperture, trace, flaw, dip, angle, coords_i, frac_type])
        # Algorithm note: see docs/METHODOLOGY.md.
    # Algorithm note: see docs/METHODOLOGY.md.
    data = sorted(data, key=lambda x: x[0])
    fracture = pandas.DataFrame(data=data,
                                columns=['center', 'normal', 'aperture', 'trace', 'flaw', 'dip', 'angle', 'pixel', 'fractype'])
    return fracture



def viewDCG2D(IMG, fracture, info):
    # Algorithm note: see docs/METHODOLOGY.md.
    ratio, start, end = info['ratio'], info['start'], info['end']
    radius, perimeter, length = info['diameter'] / 2, numpy.pi * info['diameter'], 1000 * (info['length'])
    height, width = IMG.shape[:2]
    # Algorithm note: see docs/METHODOLOGY.md.
    color_map = {
        'filled':    (0, 0, 255),  # See docs/METHODOLOGY.md.
        'unfilled':  (0, 255, 0),  # See docs/METHODOLOGY.md.
        'cemented':  (0, 255, 255),  # See docs/METHODOLOGY.md.
        'halffilled':(255, 0, 0),  # See docs/METHODOLOGY.md.
        'unknown':   (128, 128, 128)  # See docs/METHODOLOGY.md.
    }
    # Algorithm note: see docs/METHODOLOGY.md.
    image2D = cv2.copyMakeBorder(IMG, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    XY = [[radius * numpy.cos(2 * numpy.pi * x / width), radius * numpy.sin(2 * numpy.pi * x / width)] for x in
          range(width)]
    # Algorithm note: see docs/METHODOLOGY.md.
    # Algorithm note: see docs/METHODOLOGY.md.
    for i in range(len(fracture)):
        ftype = str(fracture['fractype'][i]) if 'fractype' in fracture.columns else 'unknown'
        color = color_map.get(ftype, color_map['unknown'])
        # Algorithm note: see docs/METHODOLOGY.md.
        center = numpy.array([0, 0, 1000 * (fracture['center'][i] - start)])
        x, y, z = fracture['normal'][i]
        normal = numpy.array([y, x, -z])
        tmp = sum(normal * center)
        Z = [int((tmp - normal[0] * m - normal[1] * n) / normal[2] / ratio[0]) for m, n in XY]
        ZX = [[Z[m], m] for m in range(width) if Z[m] in range(0, height)]
        for m, n in ZX:
            # Algorithm note: see docs/METHODOLOGY.md.
            image2D[m:m + 30, n:n + 30, :] = color
    # Algorithm note: see docs/METHODOLOGY.md.
    image2D = image2D[3:-3, 3:-3, :]
    # Algorithm note: see docs/METHODOLOGY.md.
    file = str(Path(info['path']) / (info['name'] + '-image2D.png'))
    save_image(file, image2D)
    return file


def fracture_parameter_table(fracture):
    """导出每个裂隙面的参数；角度转为度，法向量拆成三列。"""
    type_names = {'filled': '充填', 'unfilled': '无充填',
                  'cemented': '胶结', 'halffilled': '半充填', 'unknown': '未知'}
    table = pandas.DataFrame(index=fracture.index)
    table['裂隙编号'] = numpy.arange(1, len(fracture) + 1)
    table['裂隙类型'] = fracture['fractype'].map(type_names).fillna('未知')
    table['类型标识'] = fracture['fractype']
    table['深度(m)'] = fracture['center']
    table['开度(mm)'] = fracture['aperture']
    table['迹线长度(mm)'] = fracture['trace']
    table['开度乘迹线长度(mm²)'] = fracture['flaw']
    table['倾向(°)'] = numpy.degrees(fracture['dip'].astype(float))
    table['倾角(°)'] = numpy.degrees(fracture['angle'].astype(float))
    for axis, index in [('X', 0), ('Y', 1), ('Z', 2)]:
        table[f'法向量{axis}'] = fracture['normal'].apply(lambda value: value[index])
    table['骨架点数'] = fracture['pixel'].apply(len)
    for source, label in [('segment_name', '来源孔段'), ('segment_fracture_id', '孔段内编号'),
                          ('segment_start', '孔段起点(m)'), ('segment_end', '孔段终点(m)')]:
        if source in fracture.columns:
            table[label] = fracture[source]
    return table


def save_fracture_table(fracture, output_path):
    """保存真正的 .xlsx 工作簿，供单段输出及全孔汇总复用。"""
    try:
        from .excel_export import write_workbook
    except ImportError:
        from excel_export import write_workbook
    return write_workbook(output_path, {'裂隙参数': fracture_parameter_table(fracture)})



def cluster_image(recognition_image, original_image, *, length_m, start_depth_m,
                  diameter_mm=76.0, ratio_mm_per_pixel=None, pruning_times=1,
                  branch_length_threshold=20, clustering_mean_threshold=1.7,
                  clustering_std_threshold=1.5):
    """内存接口：返回裂隙表、绘图参数、二值图、骨架和聚类数，不写文件。"""
    for name, image in [('识别图', recognition_image), ('原图', original_image)]:
        if image is None or image.size == 0 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f'{name}必须为非空三通道 BGR 图像')
    if recognition_image.shape != original_image.shape:
        raise ValueError('识别图与原始孔段图尺寸必须一致')
    if not isinstance(pruning_times, int) or pruning_times < 0:
        raise ValueError('修剪轮数必须为非负整数')
    for value in (branch_length_threshold, clustering_mean_threshold, clustering_std_threshold):
        if not numpy.isfinite(value) or value <= 0:
            raise ValueError('骨架长度和聚类阈值必须为正数')
    info = getparameter(recognition_image, length_m=length_m, start_depth_m=start_depth_m,
                        diameter_mm=diameter_mm, ratio_mm_per_pixel=ratio_mm_per_pixel)
    masks, binary = fracget(recognition_image)
    branches = branchGeneration(binary, times=pruning_times, thresLength=branch_length_threshold)
    clusters = branchClustering(branches, thresMean=clustering_mean_threshold,
                               thresVar=clustering_std_threshold, masks=masks)
    fracture = cracksGeneration(binary, clusters, info)
    return dict(fracture=fracture, info=info, binary=binary, branches=branches,
                cluster_count=len(clusters))


def cluster_segment(recognition_image_path, segment, output_folder, *, pruning_times=1,
                    branch_length_threshold=20, clustering_mean_threshold=1.7,
                    clustering_std_threshold=1.5, save_intermediate_images=False):
    """文件接口：segment 为 segment_video 返回的单段记录。

    自动使用原始孔段作为底图，传递该段实际深度、长度和像素比例。
    返回完整裂隙表供后续统一导出 Excel 和全孔评价。
    """
    image = read_image(recognition_image_path)
    original = read_image(segment['path'])
    result = cluster_image(image, original, length_m=segment['length_m'],
                           start_depth_m=segment['start_depth_m'], diameter_mm=segment['diameter_mm'],
                           ratio_mm_per_pixel=segment.get('ratio_mm_per_pixel'),
                           pruning_times=pruning_times, branch_length_threshold=branch_length_threshold,
                           clustering_mean_threshold=clustering_mean_threshold,
                           clustering_std_threshold=clustering_std_threshold)
    destination = Path(output_folder)
    name = Path(segment['path']).stem
    result['info'].update(path=str(destination), name=name)
    result['image_path'] = viewDCG2D(original, result['fracture'], result['info'])
    if save_intermediate_images:
        save_image(destination / (name + '-binary.png'), result['binary'])
        save_image(destination / (name + '-branches.png'), result['branches'])
    result['table_path'] = str(save_fracture_table(result['fracture'], destination / (name + '-fracture_parameters.xlsx')))
    result['segment'] = dict(segment)
    return result


def main():
    if not numpy.isfinite(segment_length_m) or segment_length_m <= 0:
        raise ValueError("孔段长度必须为正数（米）")
    if not numpy.isfinite(borehole_diameter_mm) or borehole_diameter_mm <= 0:
        raise ValueError("钻孔直径必须为正数（毫米）")
    if not numpy.isfinite(start_depth_m):
        raise ValueError("起始深度必须为有限数值（米）")
    source = Path(recognition_image_path)
    image = read_image(source)
    background = read_image(original_image_path) if original_image_path else image.copy()
    if background.shape != image.shape:
        raise ValueError("原图和识别图尺寸必须一致")
    destination = Path(output_folder)
    destination.mkdir(parents=True, exist_ok=True)
    info = getparameter(image, length_m=segment_length_m, start_depth_m=start_depth_m,
                        diameter_mm=borehole_diameter_mm)
    info.update(path=str(destination), name=source.stem)
    # Algorithm note: see docs/METHODOLOGY.md.
    masks, binary = fracget(image)
    print('各类像素数:', {k: int(numpy.count_nonzero(v)) for k, v in masks.items()})
    print("开始骨架提取和修剪", flush=True)
    branches = branchGeneration(binary, times=pruning_times, thresLength=branch_length_threshold)
    print("开始类型与几何聚类", flush=True)
    clusters = branchClustering(branches, thresMean=clustering_mean_threshold,
                               thresVar=clustering_std_threshold, masks=masks)
    print(f"聚类完成，共 {len(clusters)} 组，开始参数计算", flush=True)
    fracture = cracksGeneration(binary, clusters, info)
    table_path = save_fracture_table(fracture, destination / (source.stem + '-fracture_parameters.xlsx'))
    print(f"裂隙参数表已保存到: {table_path}")
    if save_intermediate_images:
        save_image(destination / (source.stem + '-binary.png'), binary)
        save_image(destination / (source.stem + '-branches.png'), branches)
    file_2d = viewDCG2D(background, fracture, info)
    print(f"聚类数: {len(clusters)}，有效裂隙面: {len(fracture)}")
    if fracture.empty:
        print('未获得有效裂隙面，输出图仅含底图。')
    else:
        print(fracture[['center', 'aperture', 'trace', 'flaw', 'dip', 'angle', 'fractype']].to_string(index=False))
    print(f"二维数字岩心已保存到: {file_2d}")
    return fracture


if __name__ == '__main__':
    main()
