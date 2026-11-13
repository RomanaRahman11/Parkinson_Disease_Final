from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
COLUMNS=['X','Y','Z','Pressure','GripAngle','Timestamp','TestID']
FEATURE_NAMES=['n_points','x_range','y_range','path_length','median_step','median_speed','p90_speed','pressure_mean','pressure_std','pressure_median','grip_mean','grip_std']
def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(2 ** 20), b''):
            h.update(b)
    return h.hexdigest()

def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=lambda v: v.item() if hasattr(v, 'item') else str(v)), encoding='utf-8')

def assign_split(df, name, out, seed=42):
    known = {'voice': ['phon_R01_S05', 'phon_R01_S16', 'phon_R01_S17', 'phon_R01_S18', 'phon_R01_S21', 'phon_R01_S33', 'phon_R01_S37', 'phon_R01_S49'], 'spiral': ['Healthy/C_0004', 'Healthy/C_0007', 'Healthy/C_0013', 'Parkinson/P_05060003', 'Parkinson/P_11120003', 'Parkinson/P_26060007', 'Parkinson/P_27110001', 'Parkinson/P_30100001']}
    df = df.sort_values('subject_id').reset_index(drop=True).copy()
    if name in known:
        assert set(known[name]) <= set(df.subject_id), 'Known holdout IDs missing; inspect dataset before changing protocol.'
        test_ids = known[name]
        status = 'Previously evaluated holdout; retrospective reuse; no fresh-test claim'
    else:
        _, te = train_test_split(np.arange(len(df)), test_size=11, random_state=seed, stratify=df.label)
        test_ids = df.subject_id.iloc[te].tolist()
        status = 'New retrospective label-stratified MRI split; not the original label-plus-QC split'
    df['split'] = np.where(df.subject_id.isin(test_ids), 'test', 'development')
    df[['subject_id', 'label', 'split']].to_csv(out / f'{name}_split.csv', index=False)
    write_json(out / f'{name}_evaluation_status.json', {'status': status, 'external_validation': False})
    return df

def empty_features():
    return {feature: np.nan for feature in FEATURE_NAMES}

def load_trajectory(filepath):
    data = pd.read_csv(filepath, sep=';', header=None, encoding='utf-8-sig')
    if data.empty:
        raise ValueError(f'Empty trajectory: {filepath.name}')
    if data.shape[1] != 7:
        raise ValueError(f'Unexpected columns: {filepath.name}')
    data.columns = COLUMNS
    for col in COLUMNS:
        data[col] = pd.to_numeric(data[col], errors='raise')
    if not np.isfinite(data[COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError(f'Invalid numeric data: {filepath.name}')
    if not data['TestID'].isin([0, 1, 2]).all():
        raise ValueError(f'Unexpected Test ID: {filepath.name}')
    data['TestID'] = data['TestID'].astype(int)
    return data

def extract_test_features(data, test_id):
    subset = data[data['TestID'] == test_id]
    if subset.empty:
        return empty_features()
    x = subset['X'].to_numpy(dtype=float)
    y = subset['Y'].to_numpy(dtype=float)
    pressure = subset['Pressure'].to_numpy(dtype=float)
    grip = subset['GripAngle'].to_numpy(dtype=float)
    all_x = data['X'].to_numpy(dtype=float)
    all_y = data['Y'].to_numpy(dtype=float)
    all_t = data['Timestamp'].to_numpy(dtype=float)
    all_test = data['TestID'].to_numpy(dtype=int)
    dx = np.diff(all_x)
    dy = np.diff(all_y)
    dt = np.diff(all_t)
    distances = np.hypot(dx, dy)
    same_test = (all_test[:-1] == test_id) & (all_test[1:] == test_id)
    valid_time = same_test & (dt > 0)
    valid_distances = distances[valid_time]
    valid_dt = dt[valid_time]
    speeds = valid_distances / valid_dt * 1000.0 if len(valid_dt) > 0 else np.array([])
    features = {'n_points': int(len(subset)), 'x_range': float(np.ptp(x)), 'y_range': float(np.ptp(y)), 'path_length': float(np.sum(valid_distances)), 'median_step': float(np.median(valid_distances)) if len(valid_distances) > 0 else np.nan, 'median_speed': float(np.median(speeds)) if len(speeds) > 0 else np.nan, 'p90_speed': float(np.percentile(speeds, 90)) if len(speeds) > 0 else np.nan, 'pressure_mean': float(np.mean(pressure)), 'pressure_std': float(np.std(pressure)), 'pressure_median': float(np.median(pressure)), 'grip_mean': float(np.mean(grip)), 'grip_std': float(np.std(grip))}
    return features

def timestamp_quality(data):
    timestamps = data['Timestamp'].to_numpy(dtype=float)
    test_ids = data['TestID'].to_numpy(dtype=int)
    dt = np.diff(timestamps)
    same_test = test_ids[:-1] == test_ids[1:]
    return {'negative_within_test': int(np.sum((dt < 0) & same_test)), 'negative_between_tests': int(np.sum((dt < 0) & ~same_test)), 'zero_within_test': int(np.sum((dt == 0) & same_test)), 'valid_time_intervals': int(np.sum((dt > 0) & same_test))}

def trajectory_features(path):
    data = load_trajectory(path)
    record = {}
    for test_id in [0, 1, 2]:
        record.update({f'test_{test_id}_{k}': v for k, v in extract_test_features(data, test_id).items()})
    t = data.Timestamp.to_numpy(float)
    dt = np.diff(t)
    same = np.diff(data.TestID.to_numpy(int)) == 0
    positive = dt[(dt > 0) & same]
    audit = dict(n_points=len(data), median_dt=float(np.median(positive)) if len(positive) else np.nan, zero_dt_fraction=float(np.mean(dt == 0)), negative_dt_count=int((dt < 0).sum()))
    return (record, audit)

def slice_images(volume, n_slices=8):
    import torch
    import torch.nn.functional as F
    a = np.asarray(volume, np.float32)
    assert a.ndim == 3 and np.isfinite(a).all()
    images = []
    for axis in range(3):
        for frac in np.linspace(0.25, 0.75, n_slices):
            i = int(round((a.shape[axis] - 1) * frac))
            s = np.take(a, i, axis=axis).copy()
            t = torch.from_numpy(s)[None, None]
            t = F.interpolate(t, size=(224, 224), mode='bilinear', align_corners=False)[0].repeat(3, 1, 1)
            images.append(t)
    x = torch.stack(images)
    return (x - torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]) / torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]

def encode_volume(volume, encoder, device, n_slices=8):
    import torch
    slices = slice_images(volume, n_slices).to(device)
    features = []
    encoder.eval()
    with torch.inference_mode():
        for batch in slices.split(16):
            features.append(encoder(batch).float().cpu().numpy())
    z = np.concatenate(features).reshape(3, n_slices, -1)
    return np.concatenate([z.mean(axis=1).ravel(), z.std(axis=1).ravel()]).astype(np.float32)