"""全孔评价：统一数据顺序为 [深度(m), 倾向(度), 倾角(度)]。"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
try:
    from .excel_export import write_workbook
except ImportError:  # Support direct execution from the step directory.
    from excel_export import write_workbook

STANDARD_EXCEL_PATH = str(Path(__file__).resolve().parents[1] / 'data' / 'example' / 'SFGC-BX-007-III-1-9.xlsx')


def dip2normal(dip, angle):
    """dip 为倾向（正北起顺时针），angle 为倾角；返回东、北、上坐标法向量。"""
    azimuth, inclination = np.radians([dip, angle])
    return np.array([np.sin(inclination)*np.sin(azimuth),
                     np.sin(inclination)*np.cos(azimuth), np.cos(inclination)])


def _triples(data, name):
    values = np.asarray(data, dtype=float)
    if values.size == 0:
        return np.empty((0, 3), dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f'{name}必须为三列：深度、倾向、倾角')
    values = values.copy()
    # Algorithm note: see docs/METHODOLOGY.md.
    horizontal = np.isclose(values[:, 2], 0, atol=1e-10)
    values[horizontal & np.isnan(values[:, 1]), 1] = 0
    if not np.isfinite(values).all():
        raise ValueError(f'{name}存在缺失或非有限的深度/产状，不能静默剔除后评价')
    if np.any((values[:, 1] < 0) | (values[:, 1] > 360)):
        raise ValueError(f'{name}倾向应为 0–360 度')
    if np.any((values[:, 2] < 0) | (values[:, 2] > 90)):
        raise ValueError(f'{name}倾角应为 0–90 度')
    values[:, 1] %= 360
    return values


def load_standard_excel(excel_path, sheet_name=0):
    """查找实际表头，保留源 Excel 行号；跳过只有序号的空白预留行。"""
    raw = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)
    header = None
    for i in range(min(20, len(raw))):
        cells = [str(v).replace('\n', '').replace(' ', '') for v in raw.iloc[i]]
        columns = {}
        for key in ('深度', '倾向', '倾角'):
            hits = [j for j, value in enumerate(cells) if value.startswith(key)]
            if len(hits) == 1:
                columns[key] = hits[0]
        if len(columns) == 3:
            header = i
            break
    if header is None:
        raise ValueError('未找到深度、倾向、倾角表头')
    records = []
    for i in range(header+1, len(raw)):
        row = [raw.iat[i, columns[key]] for key in ('深度', '倾向', '倾角')]
        if all(pd.isna(value) for value in row):
            continue
        try:
            values = _triples([row], f'标准表第 {i+1} 行')[0]
        except (ValueError, TypeError) as exc:
            raise ValueError(f'标准表第 {i+1} 行无效: {exc}') from exc
        records.append([i+1, *values])
    result = pd.DataFrame(records, columns=['源Excel行号', '深度(m)', '倾向(°)', '倾角(°)'])
    if result.empty:
        raise ValueError('标准 Excel 没有有效裂隙标注')
    return result


def excel_to_list(excel_path, sheet_name=0, has_header=True, header_rows=1):
    """兼容旧接口；有表头时按列名识别，不再盲取第 2–4 列。"""
    if has_header:
        return load_standard_excel(excel_path, sheet_name).iloc[:, 1:4].values.tolist()
    raw = pd.read_excel(excel_path, sheet_name=sheet_name, header=None).iloc[:, :3].dropna(how='all')
    return _triples(raw.to_numpy(), '无表头输入').tolist()


def qualityCheck(input, standard, length):
    """保留三个指标各自独立的最优匹配；未匹配项为 0，分母取较大数量。"""
    if not np.isfinite(length) or length <= 0:
        raise ValueError('评价长度必须是正数（米）')
    predicted, reference = _triples(input, '预测数据'), _triples(standard, '标准数据')
    m, n = len(predicted), len(reference)
    if n == 0:
        raise ValueError('没有标准裂隙，无法计算评价指标')
    if m == 0:
        return 0.0, 0.0, 0.0
    positions = 1 - np.abs(reference[:, None, 0]-predicted[None, :, 0])/length
    # Algorithm note: see docs/METHODOLOGY.md.
    normals0 = np.array([dip2normal(d, a) for _, d, a in reference])
    normals1 = np.array([dip2normal(d, a) for _, d, a in predicted])
    angles = np.clip(np.abs(normals0 @ normals1.T), 0, 1)
    scores = []
    size = max(m, n)
    for matrix in (positions, angles, positions*angles):
        padded = np.zeros((size, size))
        padded[:n, :m] = matrix
        indices = linear_sum_assignment(-padded)
        scores.append(float(padded[indices].sum()/size))
    return tuple(scores)


def evaluate_borehole(fracture, standard_excel_path, length_m, start_depth_m=0.0, sheet_name=0):
    """评价一次整孔；不按孔段平均，不按裂隙类型额外加权。"""
    standard = load_standard_excel(standard_excel_path, sheet_name)
    predicted = _triples(np.column_stack([fracture['center'].to_numpy(dtype=float),
                         np.degrees(fracture['dip'].to_numpy(dtype=float)),
                         np.degrees(fracture['angle'].to_numpy(dtype=float))]), '全孔预测')
    reference = standard.iloc[:, 1:4].to_numpy(dtype=float)
    end = start_depth_m + length_m
    for name, values in [('标准', reference)]:
        if len(values) and np.any((values[:, 0] < start_depth_m-1e-8) | (values[:, 0] > end+1e-8)):
            raise ValueError(f'{name}深度超出全孔范围 [{start_depth_m}, {end}]，请核对数据')
    qpi, qai, cei = qualityCheck(predicted, reference, length_m)
    metrics = pd.DataFrame([dict(QPI=qpi, QAI=qai, CEI=cei,
        预测裂隙数=len(predicted), 标准裂隙数=len(reference),
        全孔范围外预测数=int(np.count_nonzero((predicted[:, 0] < start_depth_m) | (predicted[:, 0] > end))),
        **{'起始深度(m)': start_depth_m, '终止深度(m)': end, '评价长度(m)': length_m},
        状态='无预测裂隙，三个指标为0' if len(predicted)==0 else '已计算')])
    return dict(metrics=metrics, standard=standard)


def collect_borehole(segment_results, *, filter_to_segment):
    """按明确选定的孔段归属规则汇总，保留来源编号，不作相似裂隙合并。"""
    if not segment_results:
        raise ValueError('没有孔段结果，不能生成完整全孔评价')
    ordered = sorted(segment_results, key=lambda r: r['segment']['start_depth_m'])
    kept, excluded = [], []
    for index, result in enumerate(ordered):
        segment = result['segment']
        if index and not np.isclose(segment['start_depth_m'], ordered[index-1]['segment']['end_depth_m'], atol=1e-8, rtol=0):
            raise ValueError('孔段存在重叠或缺口，不能报告完整全孔评价')
        frame = result['fracture'].copy()
        frame['segment_name'] = Path(segment['path']).stem
        frame['segment_fracture_id'] = np.arange(1, len(frame)+1)
        frame['segment_start'] = segment['start_depth_m']
        frame['segment_end'] = segment['end_depth_m']
        valid = frame['center'] >= segment['start_depth_m']
        if index == len(ordered)-1:
            valid &= frame['center'] <= segment['end_depth_m']
        else:
            valid &= frame['center'] < segment['end_depth_m']
        if filter_to_segment:
            rejected = frame.loc[~valid].copy()
            rejected['reason'] = '拟合中心不在所属孔段深度范围'
            excluded.append(rejected)
            frame = frame.loc[valid]
        kept.append(frame)
    combined = pd.concat(kept, ignore_index=True).sort_values('center', kind='stable').reset_index(drop=True)
    rejected = pd.concat(excluded, ignore_index=True) if excluded else combined.iloc[:0].copy()
    return combined, rejected


def save_borehole_report(segment_results, output_folder, standard_excel_path, length_m,
                         start_depth_m=0.0, *, filter_to_segment, sheet_name=0):
    try:
        from .pic_clustering import fracture_parameter_table
    except ImportError:
        from pic_clustering import fracture_parameter_table
    ordered = sorted(segment_results, key=lambda r: r['segment']['start_depth_m'])
    if not ordered or not np.isclose(ordered[0]['segment']['start_depth_m'], start_depth_m) or not np.isclose(ordered[-1]['segment']['end_depth_m'], start_depth_m+length_m):
        raise ValueError('孔段结果未覆盖配置的整个孔长')
    combined, excluded = collect_borehole(ordered, filter_to_segment=filter_to_segment)
    evaluation = evaluate_borehole(combined, standard_excel_path, length_m, start_depth_m, sheet_name)
    method = pd.DataFrame([
        ['标准来源', Path(standard_excel_path).name],
        ['数据顺序与单位', '深度(m)、倾向(度)、倾角(度)；倾向为正北起顺时针'],
        ['QPI', '1 - 深度差绝对值 / 全孔长度'],
        ['QAI', '两个单位法向量点积的绝对值'],
        ['CEI', '每对裂隙 QPI × QAI 后再独立最优匹配'],
        ['匹配方法', '三个指标各自最优匹配；未匹配记0；除以预测与标准数量较大值'],
        ['孔段归属', '左闭右开，末段含终点' if filter_to_segment else '保留全部拟合结果'],
        ['合并规则', '不根据相近深度自动合并裂隙'],
        ['范围外预测', '保留并参与评价；QPI按原公式计算，不截断负值'],
        ['筛除数量', len(excluded)],
        ['类型', '裂隙类型保留在参数表中，不参与这三个评价指标计算'],
    ], columns=['项目', '说明'])
    sheets = {'评价指标': evaluation['metrics'], '全孔裂隙参数': fracture_parameter_table(combined),
              '标准裂隙': evaluation['standard'], '计算说明': method}
    if len(excluded):
        excluded_table = fracture_parameter_table(excluded)
        excluded_table['筛除原因'] = excluded['reason'].values
        sheets['孔段范围外裂隙'] = excluded_table
    path = write_workbook(Path(output_folder)/'borehole_evaluation.xlsx', sheets)
    return dict(path=str(path), fracture=combined, excluded=excluded, **evaluation)
