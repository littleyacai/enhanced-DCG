import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skan import Skeleton, summarize
import pandas
from pathlib import Path

# Algorithm note: see docs/METHODOLOGY.md.

# Algorithm note: see docs/METHODOLOGY.md.
input_folder = str(Path(__file__).resolve().parents[1] / "outputs" / "eliminate")
output_folder = str(Path(__file__).resolve().parents[1] / "outputs" / "extension")
reference_point_idx = 50  # See docs/METHODOLOGY.md.
num_expand_points = 10  # See docs/METHODOLOGY.md.


# Algorithm note: see docs/METHODOLOGY.md.
def extract_color_regions(img: np.ndarray):
    """
    提取红色（充填）、绿色（无充填）和黄色（胶结）连通域区域。

    参数:
        img: 彩色图像 (BGR)，类型为 np.ndarray

    返回:
        red_set: [(label_id, [(y1, x1), (y2, x2), ...]), ...]
        green_set: [(label_id, [(y1, x1), (y2, x2), ...]), ...]  # See docs/METHODOLOGY.md.
        yellow_set: [(label_id, [(y1, x1), (y2, x2), ...]), ...]  # See docs/METHODOLOGY.md.
    """

    # Algorithm note: see docs/METHODOLOGY.md.
    lower_red = np.array([0, 0, 120])  # See docs/METHODOLOGY.md.
    upper_red = np.array([10, 10, 140])
    lower_green = np.array([0, 120, 0])  # See docs/METHODOLOGY.md.
    upper_green = np.array([10, 140, 10])
    lower_yellow = np.array([0, 120, 120])  # See docs/METHODOLOGY.md.
    upper_yellow = np.array([10, 140, 140])

    # Algorithm note: see docs/METHODOLOGY.md.
    red_mask = cv2.inRange(img, lower_red, upper_red)
    num_labels_red, labels_red = cv2.connectedComponents(red_mask)

    red_set = []
    for label_id in range(1, num_labels_red):  # See docs/METHODOLOGY.md.
        coords = np.argwhere(labels_red == label_id)
        red_set.append((label_id, coords.tolist()))

    # Algorithm note: see docs/METHODOLOGY.md.
    green_mask = cv2.inRange(img, lower_green, upper_green)
    num_labels_green, labels_green = cv2.connectedComponents(green_mask)

    green_set = []
    for label_id in range(1, num_labels_green):
        coords = np.argwhere(labels_green == label_id)
        green_set.append((label_id, coords.tolist()))

    # Algorithm note: see docs/METHODOLOGY.md.
    yellow_mask = cv2.inRange(img, lower_yellow, upper_yellow)
    num_labels_yellow, labels_yellow = cv2.connectedComponents(yellow_mask)

    yellow_set = []
    for label_id in range(1, num_labels_yellow):
        coords = np.argwhere(labels_yellow == label_id)
        yellow_set.append((label_id, coords.tolist()))

    return red_set, green_set, yellow_set


# Algorithm note: see docs/METHODOLOGY.md.
def get_direction(coords1, coords2):
    vectors = coords1 - coords2
    norm = np.linalg.norm(vectors)
    normalized_vectors = vectors / (norm + 1e-8)  # See docs/METHODOLOGY.md.
    return normalized_vectors


# Algorithm note: see docs/METHODOLOGY.md.
def point_width(bin_image, point, direction_vector):
    # Algorithm note: see docs/METHODOLOGY.md.
    perp_vector = np.array([-direction_vector[1], direction_vector[0]])

    # Algorithm note: see docs/METHODOLOGY.md.
    y, x = int(round(point[0])), int(round(point[1]))

    # Algorithm note: see docs/METHODOLOGY.md.
    width_pos = 0
    step = 1
    while True:
        ny = y + int(round(perp_vector[0] * step))
        nx = x + int(round(perp_vector[1] * step))
        # Algorithm note: see docs/METHODOLOGY.md.
        if not (0 <= ny < bin_image.shape[0] and 0 <= nx < bin_image.shape[1]):
            break
        # Algorithm note: see docs/METHODOLOGY.md.
        if bin_image[ny, nx] == 0:
            break

        width_pos = step
        step += 1

    # Algorithm note: see docs/METHODOLOGY.md.
    width_neg = 0
    step = 1
    while True:
        ny = y - int(round(perp_vector[0] * step))
        nx = x - int(round(perp_vector[1] * step))

        if not (0 <= ny < bin_image.shape[0] and 0 <= nx < bin_image.shape[1]):
            break

        if bin_image[ny, nx] == 0:
            break

        width_neg = step
        step += 1

    # Algorithm note: see docs/METHODOLOGY.md.
    return width_pos + width_neg

# Algorithm note: see docs/METHODOLOGY.md.
def extend_image(image, reference_point_idx=50, num_expand_points=5):
    """
    处理图像的主函数，允许指定参考点位置

    参数:
        image: 三通道 BGR 图像数组，不修改传入数组
        reference_point_idx: 用于计算方向和宽度的参考点索引（从端点起算）
        num_expand_points: 扩展点的数量
    """
    if not isinstance(reference_point_idx, int) or reference_point_idx <= 0:
        raise ValueError('参考点索引必须为正整数')
    if not isinstance(num_expand_points, int) or num_expand_points < 0:
        raise ValueError('最大扩展点数必须为非负整数')
    if image is None or image.size == 0 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('输入必须为非空三通道 BGR 图像')
    image = image.copy()
    bin_image = np.any(image != [0, 0, 0], axis=2).astype(np.uint8)

    # Algorithm note: see docs/METHODOLOGY.md.
    skeleton = skeletonize(bin_image)
    # Algorithm note: see docs/METHODOLOGY.md.
    neighbors = cv2.filter2D(skeleton.astype(np.uint8), -1, np.ones((3, 3), np.uint8),
                             borderType=cv2.BORDER_CONSTANT)
    if not np.any(skeleton & (neighbors > 1)):
        return image.copy(), pandas.DataFrame()
    skeleton_obj = Skeleton(skeleton)
    skeleton_properties = summarize(skeleton_obj, separator='-')

    # Algorithm note: see docs/METHODOLOGY.md.
    allPoints = list(skeleton_properties['node-id-src']) + list(skeleton_properties['node-id-dst'])
    intpoints = list(set([n for n in allPoints if allPoints.count(n) > 1]))
    endpoints = list(set(allPoints) - set(intpoints))

    paths = skeleton_obj.paths_list()

    # Algorithm note: see docs/METHODOLOGY.md.
    endpoint_to_reference_point = {}

    for i, path in enumerate(paths):
        src_id = skeleton_properties['node-id-src'][i]
        dst_id = skeleton_properties['node-id-dst'][i]

        if src_id in endpoints and len(path) > reference_point_idx:
            # Algorithm note: see docs/METHODOLOGY.md.
            reference_point = path[reference_point_idx]
            endpoint_to_reference_point[src_id] = reference_point

        if dst_id in endpoints and len(path) > reference_point_idx:
            # Algorithm note: see docs/METHODOLOGY.md.
            reference_point = path[::-1][reference_point_idx]
            endpoint_to_reference_point[dst_id] = reference_point

    # Algorithm note: see docs/METHODOLOGY.md.
    data_list = []
    for endpoint_id in endpoints:
        # Algorithm note: see docs/METHODOLOGY.md.
        endpoint_coord = skeleton_obj.coordinates[endpoint_id]  # See docs/METHODOLOGY.md.

        # Algorithm note: see docs/METHODOLOGY.md.
        if endpoint_id in endpoint_to_reference_point:
            reference_point_coord = skeleton_obj.coordinates[endpoint_to_reference_point.get(endpoint_id)]

            # Algorithm note: see docs/METHODOLOGY.md.
            direction = get_direction(np.array(endpoint_coord), np.array(reference_point_coord))

            # Algorithm note: see docs/METHODOLOGY.md.
            width = point_width(bin_image, reference_point_coord, direction)

            data_list.append({
                '端点ID': endpoint_id,
                '端点Y坐标': endpoint_coord[0],
                '端点X坐标': endpoint_coord[1],
                '参考点Y坐标': reference_point_coord[0],
                '参考点X坐标': reference_point_coord[1],
                '方向Y分量': direction[0],
                '方向X分量': direction[1],
                '参考点宽度': width
            })

    # Algorithm note: see docs/METHODOLOGY.md.
    df = pandas.DataFrame(data_list)
    if df.empty:
        return image.copy(), df

    # Algorithm note: see docs/METHODOLOGY.md.
    df = df[df['参考点宽度'] <= 100].reset_index(drop=True)

    # Algorithm note: see docs/METHODOLOGY.md.
    expansion_sets = {}

    # Algorithm note: see docs/METHODOLOGY.md.
    for index, row in df.iterrows():
        endpoint_id = int(row['端点ID'])
        # Algorithm note: see docs/METHODOLOGY.md.
        x_endpoint, y_endpoint = row['端点X坐标'], row['端点Y坐标']

        # Algorithm note: see docs/METHODOLOGY.md.
        dx, dy = row['方向X分量'], row['方向Y分量']

        # Algorithm note: see docs/METHODOLOGY.md.
        width = row['参考点宽度']

        expansion_sets[endpoint_id] = []
        # Algorithm note: see docs/METHODOLOGY.md.
        for k in range(1, num_expand_points + 1):
            # Algorithm note: see docs/METHODOLOGY.md.
            new_x = int(round(x_endpoint + k * dx * width / 2))
            new_y = int(round(y_endpoint + k * dy * width / 2))
            radius = int(round(width / 2))

            # Algorithm note: see docs/METHODOLOGY.md.
            if 0 <= new_x < bin_image.shape[1] and 0 <= new_y < bin_image.shape[0]:
                # Algorithm note: see docs/METHODOLOGY.md.
                expansion_sets[endpoint_id].append((new_x, new_y, int(round(width / 2))))

    # Algorithm note: see docs/METHODOLOGY.md.
    # Algorithm note: see docs/METHODOLOGY.md.
    temp_image = bin_image.copy()

    # Algorithm note: see docs/METHODOLOGY.md.
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_image, connectivity=8)

    # Algorithm note: see docs/METHODOLOGY.md.
    endpoints_to_remove = []
    for index, expansion_set in expansion_sets.items():
        # Algorithm note: see docs/METHODOLOGY.md.
        endpoint_x = df.loc[df['端点ID'] == index, '端点X坐标'].values[0]
        endpoint_y = df.loc[df['端点ID'] == index, '端点Y坐标'].values[0]
        current_component_label = labels[endpoint_y, endpoint_x]  # See docs/METHODOLOGY.md.

        # Algorithm note: see docs/METHODOLOGY.md.
        endpoint_mask = np.zeros_like(bin_image, dtype=np.uint8)
        for (new_x, new_y, radius) in expansion_set:
            cv2.circle(endpoint_mask, (new_x, new_y), radius, 255, -1)

        # Algorithm note: see docs/METHODOLOGY.md.
        other_components_mask = np.zeros_like(bin_image, dtype=np.uint8)
        # Algorithm note: see docs/METHODOLOGY.md.
        for label in range(1, num_labels):  # See docs/METHODOLOGY.md.
            if label != current_component_label:  # See docs/METHODOLOGY.md.
                other_components_mask[labels == label] = 255
        # Algorithm note: see docs/METHODOLOGY.md.
        for other_index, other_expansion_set in expansion_sets.items():
            if other_index == index:
                continue  # See docs/METHODOLOGY.md.
            # Algorithm note: see docs/METHODOLOGY.md.
            other_x = df.loc[df['端点ID'] == other_index, '端点X坐标'].values[0]
            other_y = df.loc[df['端点ID'] == other_index, '端点Y坐标'].values[0]
            other_component_label = labels[other_y, other_x]
            # Algorithm note: see docs/METHODOLOGY.md.
            if other_component_label != current_component_label:
                for (new_x, new_y, radius) in other_expansion_set:
                    cv2.circle(other_components_mask, (new_x, new_y), radius, 255, -1)

        # Algorithm note: see docs/METHODOLOGY.md.
        intersection = cv2.bitwise_and(endpoint_mask, other_components_mask)
        if not np.any(intersection):  # See docs/METHODOLOGY.md.
            endpoints_to_remove.append(index)

    # Algorithm note: see docs/METHODOLOGY.md.
    df = df[~df['端点ID'].isin(endpoints_to_remove)].reset_index(drop=True)

    # Algorithm note: see docs/METHODOLOGY.md.
    # Algorithm note: see docs/METHODOLOGY.md.
    bin_image_copy = bin_image.copy()

    # Algorithm note: see docs/METHODOLOGY.md.
    df['拓展次数'] = 0

    # Algorithm note: see docs/METHODOLOGY.md.
    for index, row in df.iterrows():
        endpoint_id = int(row['端点ID'])
        x_endpoint, y_endpoint = int(row['端点X坐标']), int(row['端点Y坐标'])
        dx, dy = row['方向X分量'], row['方向Y分量']
        width = row['参考点宽度']

        # Algorithm note: see docs/METHODOLOGY.md.
        temp_image = bin_image.copy()
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_image, connectivity=8)
        component_label = labels[y_endpoint, x_endpoint]  # See docs/METHODOLOGY.md.
        temp_image[labels == component_label] = 0  # See docs/METHODOLOGY.md.

        # Algorithm note: see docs/METHODOLOGY.md.
        temp1 = temp_image.copy()

        for other_index, other_row in df.iterrows():
            if other_index == index:
                continue  # See docs/METHODOLOGY.md.
            ox, oy = other_row['端点X坐标'], other_row['端点Y坐标']
            for k in range(1, num_expand_points + 1):
                new_x = int(round(ox + k * other_row['方向X分量'] * other_row['参考点宽度'] / 2))
                new_y = int(round(oy + k * other_row['方向Y分量'] * other_row['参考点宽度'] / 2))
                radius = int(round(other_row['参考点宽度'] / 2))
                if 0 <= new_x < bin_image.shape[1] and 0 <= new_y < bin_image.shape[0]:
                    cv2.circle(temp1, (new_x, new_y), radius, 255, -1)

        # Algorithm note: see docs/METHODOLOGY.md.
        temp1_skeleton = skeletonize(temp1)

        expansion_mask = np.zeros_like(bin_image, dtype=np.uint8)

        # Algorithm note: see docs/METHODOLOGY.md.
        expansion_count = 0
        for k in range(0, num_expand_points + 1):
            new_x = int(round(x_endpoint + k * dx * width / 2))
            new_y = int(round(y_endpoint + k * dy * width / 2))
            radius = int(round(width / 2))

            if not (0 <= new_x < bin_image.shape[1] and 0 <= new_y < bin_image.shape[0]):
                break  # See docs/METHODOLOGY.md.

            # Algorithm note: see docs/METHODOLOGY.md.
            cv2.circle(expansion_mask, (new_x, new_y), radius, 255, -1)

            # Algorithm note: see docs/METHODOLOGY.md.
            expansion_skeleton = skeletonize(expansion_mask)
            # Algorithm note: see docs/METHODOLOGY.md.
            intersection = np.logical_and(temp1_skeleton, expansion_skeleton)
            if np.any(intersection):
                break  # See docs/METHODOLOGY.md.

            # Algorithm note: see docs/METHODOLOGY.md.
            expansion_count += 1

            cv2.circle(expansion_mask, (new_x, new_y), radius, 255, -1)  # See docs/METHODOLOGY.md.

        # Algorithm note: see docs/METHODOLOGY.md.
        if expansion_count == num_expand_points + 1:  # See docs/METHODOLOGY.md.
            # Algorithm note: see docs/METHODOLOGY.md.
            actual_expansion_count = 0
            temp_expansion_mask = np.zeros_like(bin_image, dtype=np.uint8)

            for k in range(0, num_expand_points + 1):
                new_x = int(round(x_endpoint + k * dx * width / 2))
                new_y = int(round(y_endpoint + k * dy * width / 2))
                radius = int(round(width / 2))

                if not (0 <= new_x < bin_image.shape[1] and 0 <= new_y < bin_image.shape[0]):
                    break  # See docs/METHODOLOGY.md.

                # Algorithm note: see docs/METHODOLOGY.md.
                cv2.circle(temp_expansion_mask, (new_x, new_y), radius, 255, -1)

                # Algorithm note: see docs/METHODOLOGY.md.
                if np.any(np.logical_and(temp_expansion_mask, temp_image)):
                    actual_expansion_count = k
                    break

            # Algorithm note: see docs/METHODOLOGY.md.
            df.at[index, '拓展次数'] = actual_expansion_count
        else:
            df.at[index, '拓展次数'] = expansion_count

    bin_image_copy1 = cv2.cvtColor((bin_image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    # Algorithm note: see docs/METHODOLOGY.md.
    for index, row in df.iterrows():
        # Algorithm note: see docs/METHODOLOGY.md.
        x_endpoint, y_endpoint = row['端点X坐标'], row['端点Y坐标']

        # Algorithm note: see docs/METHODOLOGY.md.
        dx, dy = row['方向X分量'], row['方向Y分量']

        # Algorithm note: see docs/METHODOLOGY.md.
        width = row['参考点宽度']

        # Algorithm note: see docs/METHODOLOGY.md.
        for k in range(1, int(row['拓展次数']) + 1):
            # Algorithm note: see docs/METHODOLOGY.md.
            new_x = int(round(x_endpoint + k * dx * width / 2))
            new_y = int(round(y_endpoint + k * dy * width / 2))
            radius = int(round(width / 2))

            # Algorithm note: see docs/METHODOLOGY.md.
            if 0 <= new_x < bin_image.shape[1] and 0 <= new_y < bin_image.shape[0]:
                # Algorithm note: see docs/METHODOLOGY.md.
                cv2.circle(bin_image_copy1, (new_x, new_y), radius, (255, 255, 255), -1)

    # Algorithm note: see docs/METHODOLOGY.md.

    # Algorithm note: see docs/METHODOLOGY.md.
    red_mask = cv2.inRange(image, np.array([0, 0, 120]), np.array([10, 10, 140]))
    green_mask = cv2.inRange(image, np.array([0, 120, 0]), np.array([10, 140, 10]))
    yellow_mask = cv2.inRange(image, np.array([0, 120, 120]), np.array([10, 140, 140]))

    # Algorithm note: see docs/METHODOLOGY.md.
    color_result = image.copy()

    # Algorithm note: see docs/METHODOLOGY.md.
    extension_mask_all = np.zeros_like(bin_image, dtype=np.uint8)

    # Algorithm note: see docs/METHODOLOGY.md.
    for index, row in df.iterrows():
        x, y = int(row['端点X坐标']), int(row['端点Y坐标'])
        dx, dy = row['方向X分量'], row['方向Y分量']
        width = row['参考点宽度']
        times = int(row['拓展次数'])
        for k in range(0, times + 1):
            new_x = int(round(x + k * dx * width / 2))
            new_y = int(round(y + k * dy * width / 2))
            radius = int(round(width / 2))
            if 0 <= new_x < bin_image.shape[1] and 0 <= new_y < bin_image.shape[0]:
                cv2.circle(extension_mask_all, (new_x, new_y), radius, 255, -1)

    # Algorithm note: see docs/METHODOLOGY.md.
    num_labels_ext, labels_ext = cv2.connectedComponents(extension_mask_all)

    # Algorithm note: see docs/METHODOLOGY.md.
    red_set, green_set, yellow_set = extract_color_regions(image)

    # Algorithm note: see docs/METHODOLOGY.md.
    red_regions = []
    for label_id, coords in red_set:
        red_regions.append(np.array(coords))

    green_regions = []
    for label_id, coords in green_set:
        green_regions.append(np.array(coords))

    yellow_regions = []
    for label_id, coords in yellow_set:
        yellow_regions.append(np.array(coords))

    # Algorithm note: see docs/METHODOLOGY.md.
    connected_red_ext_labels = []  # See docs/METHODOLOGY.md.
    connected_green_ext_labels = []  # See docs/METHODOLOGY.md.
    connected_yellow_ext_labels = []  # See docs/METHODOLOGY.md.
    connected_mixed_ext_labels = []  # See docs/METHODOLOGY.md.

    for ext_label in range(1, num_labels_ext):
        ext_coords = np.argwhere(labels_ext == ext_label)

        # Algorithm note: see docs/METHODOLOGY.md.
        ext_mask = np.zeros_like(bin_image, dtype=np.uint8)
        for y, x in ext_coords:
            ext_mask[y, x] = 1

        # Algorithm note: see docs/METHODOLOGY.md.
        connected_red_regions = []
        for i, red_region in enumerate(red_regions):
            red_mask_i = np.zeros_like(bin_image, dtype=np.uint8)
            for y, x in red_region:
                red_mask_i[y, x] = 1

            if np.any(np.logical_and(ext_mask, red_mask_i)):
                connected_red_regions.append(i)

        # Algorithm note: see docs/METHODOLOGY.md.
        connected_green_regions = []
        for i, green_region in enumerate(green_regions):
            green_mask_i = np.zeros_like(bin_image, dtype=np.uint8)
            for y, x in green_region:
                green_mask_i[y, x] = 1

            if np.any(np.logical_and(ext_mask, green_mask_i)):
                connected_green_regions.append(i)

        # Algorithm note: see docs/METHODOLOGY.md.
        connected_yellow_regions = []
        for i, yellow_region in enumerate(yellow_regions):
            yellow_mask_i = np.zeros_like(bin_image, dtype=np.uint8)
            for y, x in yellow_region:
                yellow_mask_i[y, x] = 1

            if np.any(np.logical_and(ext_mask, yellow_mask_i)):
                connected_yellow_regions.append(i)

        # Algorithm note: see docs/METHODOLOGY.md.
        if len(connected_red_regions) >= 2 and len(connected_green_regions) == 0 and len(connected_yellow_regions) == 0:
            # Algorithm note: see docs/METHODOLOGY.md.
            connected_red_ext_labels.append(ext_label)
        elif len(connected_green_regions) >= 2 and len(connected_red_regions) == 0 and len(
                connected_yellow_regions) == 0:
            # Algorithm note: see docs/METHODOLOGY.md.
            connected_green_ext_labels.append(ext_label)
        elif len(connected_yellow_regions) >= 2 and len(connected_red_regions) == 0 and len(
                connected_green_regions) == 0:
            # Algorithm note: see docs/METHODOLOGY.md.
            connected_yellow_ext_labels.append(ext_label)
        elif len(connected_red_regions) >= 1 and (
                len(connected_green_regions) >= 1 or len(connected_yellow_regions) >= 1):
            # Algorithm note: see docs/METHODOLOGY.md.
            connected_mixed_ext_labels.append(
                (ext_label, connected_red_regions, connected_green_regions, connected_yellow_regions))

    # Algorithm note: see docs/METHODOLOGY.md.
    for ext_label in connected_red_ext_labels:
        ext_coords = np.argwhere(labels_ext == ext_label)
        for y, x in ext_coords:
            # Algorithm note: see docs/METHODOLOGY.md.
            if red_mask[y, x] == 0 and green_mask[y, x] == 0 and yellow_mask[y, x] == 0:
                # Algorithm note: see docs/METHODOLOGY.md.
                color_result[y, x] = [0, 0, 128]

    # Algorithm note: see docs/METHODOLOGY.md.
    for ext_label in connected_green_ext_labels:
        ext_coords = np.argwhere(labels_ext == ext_label)
        for y, x in ext_coords:
            # Algorithm note: see docs/METHODOLOGY.md.
            if red_mask[y, x] == 0 and green_mask[y, x] == 0 and yellow_mask[y, x] == 0:
                # Algorithm note: see docs/METHODOLOGY.md.
                if not np.array_equal(color_result[y, x], [0, 0, 128]):
                    # Algorithm note: see docs/METHODOLOGY.md.
                    color_result[y, x] = [0, 128, 0]

    # Algorithm note: see docs/METHODOLOGY.md.
    for ext_label in connected_yellow_ext_labels:
        ext_coords = np.argwhere(labels_ext == ext_label)
        for y, x in ext_coords:
            # Algorithm note: see docs/METHODOLOGY.md.
            if red_mask[y, x] == 0 and green_mask[y, x] == 0 and yellow_mask[y, x] == 0:
                # Algorithm note: see docs/METHODOLOGY.md.
                if not np.array_equal(color_result[y, x], [0, 0, 128]) and not np.array_equal(color_result[y, x],
                                                                                              [0, 128, 0]):
                    # Algorithm note: see docs/METHODOLOGY.md.
                    color_result[y, x] = [0, 128, 128]

    # Algorithm note: see docs/METHODOLOGY.md.
    for ext_label, red_region_ids, green_region_ids, yellow_region_ids in connected_mixed_ext_labels:
        # Algorithm note: see docs/METHODOLOGY.md.
        red_area = 0
        for red_id in red_region_ids:
            red_area += len(red_regions[red_id])

        green_area = 0
        for green_id in green_region_ids:
            green_area += len(green_regions[green_id])

        yellow_area = 0
        for yellow_id in yellow_region_ids:
            yellow_area += len(yellow_regions[yellow_id])

        total_area = red_area + green_area + yellow_area
        red_ratio = red_area / total_area if total_area > 0 else 0
        green_ratio = green_area / total_area if total_area > 0 else 0
        yellow_ratio = yellow_area / total_area if total_area > 0 else 0

        # Algorithm note: see docs/METHODOLOGY.md.
        connected_points = []

        # Algorithm note: see docs/METHODOLOGY.md.
        ext_coords = np.argwhere(labels_ext == ext_label)
        connected_points.extend([(y, x) for y, x in ext_coords])

        # Algorithm note: see docs/METHODOLOGY.md.
        for red_id in red_region_ids:
            connected_points.extend([(y, x) for y, x in red_regions[red_id]])

        # Algorithm note: see docs/METHODOLOGY.md.
        for green_id in green_region_ids:
            connected_points.extend([(y, x) for y, x in green_regions[green_id]])

        # Algorithm note: see docs/METHODOLOGY.md.
        for yellow_id in yellow_region_ids:
            connected_points.extend([(y, x) for y, x in yellow_regions[yellow_id]])

        # Algorithm note: see docs/METHODOLOGY.md.
        color = None

        # Algorithm note: see docs/METHODOLOGY.md.
        if len(red_region_ids) > 0 and len(green_region_ids) > 0 and len(yellow_region_ids) == 0:
            # Algorithm note: see docs/METHODOLOGY.md.
            if red_ratio > 0.7:  # See docs/METHODOLOGY.md.
                color = [0, 0, 128]  # See docs/METHODOLOGY.md.
            elif red_ratio < 0.3:  # See docs/METHODOLOGY.md.
                color = [0, 128, 0]  # See docs/METHODOLOGY.md.
            else:  # See docs/METHODOLOGY.md.
                color = [255, 0, 0]  # See docs/METHODOLOGY.md.
        else:
            # Algorithm note: see docs/METHODOLOGY.md.
            if red_ratio >= green_ratio and red_ratio >= yellow_ratio:
                color = [0, 0, 128]  # See docs/METHODOLOGY.md.
            elif green_ratio >= yellow_ratio:
                color = [0, 128, 0]  # See docs/METHODOLOGY.md.
            else:
                color = [0, 128, 128]  # See docs/METHODOLOGY.md.

        # Algorithm note: see docs/METHODOLOGY.md.
        for y, x in connected_points:
            if 0 <= y < color_result.shape[0] and 0 <= x < color_result.shape[1]:
                color_result[y, x] = color

    # Algorithm note: see docs/METHODOLOGY.md.
    return color_result, df


def process_image(image_path, reference_point_idx=50, num_expand_points=5, output_path=None):
    """文件接口：读取、重连，可选保存；返回图像和端点数据。"""
    image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'无法读取图片: {image_path}')
    result, endpoints = extend_image(image, reference_point_idx, num_expand_points)
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        ok, encoded = cv2.imencode(destination.suffix, result)
        if not ok:
            raise RuntimeError(f'图片编码失败: {destination}')
        encoded.tofile(str(destination))
    return result, endpoints


def main():
    if not isinstance(reference_point_idx, int) or reference_point_idx <= 0:
        raise ValueError("参考点索引必须是大于 0 的整数")
    if not isinstance(num_expand_points, int) or num_expand_points < 0:
        raise ValueError("最大扩展点数必须是非负整数")
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
            color_result, df = process_image(image_path, reference_point_idx, num_expand_points)
            # Algorithm note: see docs/METHODOLOGY.md.
            output_path = destination / f"{image_path.name}_extension.png"
            success, encoded = cv2.imencode(".png", color_result)
            if not success:
                raise RuntimeError("结果图片编码失败")
            encoded.tofile(str(output_path))
            succeeded += 1
            print(f"[{index}/{len(image_paths)}] 已保存: {output_path}")
        except Exception as exc:
            print(f"[{index}/{len(image_paths)}] 处理失败: {image_path.name}: {exc}")
    print(f"处理完成：成功 {succeeded} 张，失败 {len(image_paths) - succeeded} 张。保存位置: {destination}")


if __name__ == "__main__":
    main()
