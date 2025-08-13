import os
import pandas as pd
import numpy as np


def clean_mouse_data(file_path):
    """
    Load and clean raw mouse event data from a file.

    - Removes duplicate rows.
    - Excludes events where `button` equals 'Scroll'.
    - Drops the 'record timestamp' column.

    Parameters
    ----------
    file_path : str
        Path to the CSV file containing raw mouse data.

    Returns
    -------
    pandas.DataFrame
        Cleaned mouse event data.
    """
    
    data = pd.read_csv(file_path)
    data_cleaned = data.drop_duplicates()
    data_cleaned = data_cleaned[data_cleaned['button'] != 'Scroll']
    data_cleaned.drop('record timestamp', axis=1, inplace=True)
    return data_cleaned


def segment_mouse_actions(data, time_threshold=10.0):
    """
    Segment mouse data into discrete actions.

    Actions are split into:
    - 'PC' (Point Click)
    - 'MM' (Mouse Move)
    - 'DD' (Drag & Drop)

    Rules:
    - Actions with fewer than 4 points are ignored.
    - Time gaps between consecutive points greater than `time_threshold`
    start a new segment.

    Parameters
    ----------
    data : pandas.DataFrame
        Cleaned mouse event data.
    time_threshold : float, optional
        Maximum time (in ms) allowed between points before splitting, by default 10.0.

    Returns
    -------
    list of (pandas.DataFrame, str)
        List of (segment_data, action_type) tuples.
    """

    segments = []
    current_segment = []
    for index, row in data.iterrows():
        current_segment.append(row)
        if row['state'] == 'Released':
            if len(current_segment) > 1 and current_segment[-2]['state'] == 'Pressed':
                action_type = 'PC'  # Point Click
                if len(current_segment) >= 4:
                    segments.append((current_segment.copy(), action_type))
            elif len(current_segment) > 1:
                mm_list = []
                dd_list = []
                for i in range(len(current_segment) - 1, -1, -1):
                    if current_segment[i]['state'] == 'Pressed':
                        mm_list = current_segment[0:i]
                        dd_list = current_segment[i:len(current_segment)]
                        break
                if len(mm_list) >= 4:
                    segments.append((mm_list, 'MM')) # Mouse Move
                if len(dd_list) >= 4:
                    segments.append((dd_list, 'DD')) # Drag & Drop
            current_segment = []
    final_segments = []
    for segment, action_type in segments:
        temp_segment = []
        for i in range(len(segment)):
            temp_segment.append(segment[i])
            if i != len(segment) - 1:
                if segment[i + 1]['client timestamp'] - segment[i]['client timestamp'] > time_threshold:
                    if len(temp_segment) >= 4:
                        final_segments.append((temp_segment.copy(), 'MM'))
                    temp_segment = []
        if len(temp_segment) >= 4:
            final_segments.append((temp_segment.copy(), action_type))
    final_segments_df = [(pd.DataFrame(seg), action_type) for seg, action_type in final_segments]
    return final_segments_df

def replace_zeros(dt, j=0):
    """
    Replace zero values in an array with interpolated values from neighbors.

    If any zeros remain after a pass, the function recurses until all zeros are replaced.

    Parameters
    ----------
    dt : array-like
        Input array of time differences.
    j : int, optional
        Recursion depth counter (unused in computation).

    Returns
    -------
    numpy.ndarray
        Array with zero values replaced.
    """

    dt = np.array(dt)
    n = len(dt)
    for i in range(n):
        if dt[i] == 0.0:
            if i == 0:
                dt[i] = dt[i + 1]
            elif i == n - 1:
                dt[i] = dt[i - 1]
            else:
                dt[i] = (dt[i - 1] + dt[i + 1]) / 2
    while np.any(dt == 0):
        dt = replace_zeros(dt, j)
    return dt


def compute_features(segment):
    """
    Compute kinematic features from a single mouse action segment.

    Extracts features such as:
    - Velocity, acceleration, jerk
    - Angular velocity, curvature
    - Total time, path length, straightness
    - Direction, point count, total angle
    - Various statistical metrics (mean, std, min, max)

    If all time differences are zero, returns None.

    Parameters
    ----------
    segment : pandas.DataFrame
        Mouse action segment.

    Returns
    -------
    dict or None
        Dictionary of computed features, or None if invalid.
    """

    x = segment['x'].values
    y = segment['y'].values
    t = segment['client timestamp'].values
    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.diff(t)
    if np.all(dt == 0):
        return None
    dt = replace_zeros(dt)
    velocities_x = dx / dt
    velocities_y = dy / dt
    speeds = np.sqrt(velocities_x**2 + velocities_y**2)
    accelerations = np.diff(speeds) / dt[1:]
    jerks = np.diff(accelerations) / dt[2:]
    angles = np.arctan2(dy, dx)
    angular_velocities = np.diff(angles) / dt[1:]
    curvatures = np.divide(angular_velocities, speeds[1:], out=np.zeros_like(angular_velocities), where=speeds[1:]!=0)
    total_time = t[-1] - t[0]
    trajectory_length = np.sum(np.sqrt(dx**2 + dy**2))
    straight_distance = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)
    straightness = straight_distance / trajectory_length if trajectory_length > 0 else 0
    direction = np.arctan2(y[-1] - y[0], x[-1] - x[0])
    point_count = len(x)
    total_angle = np.sum(np.abs(np.diff(angles)))
    initial_acceleration = accelerations[0] if len(accelerations) > 0 else 0
    features = {
        'average_speed_x': np.mean(velocities_x),
        'std_dev_speed_x': np.std(velocities_x),
        'min_speed_x': np.min(velocities_x),
        'max_speed_x': np.max(velocities_x),
        'average_speed_y': np.mean(velocities_y),
        'std_dev_speed_y': np.std(velocities_y),
        'min_speed_y': np.min(velocities_y),
        'max_speed_y': np.max(velocities_y),
        'average_speed': np.mean(speeds),
        'std_dev_speed': np.std(speeds),
        'min_speed': np.min(speeds),
        'max_speed': np.max(speeds),
        'average_acceleration': np.mean(accelerations),
        'std_dev_acceleration': np.std(accelerations),
        'min_acceleration': np.min(accelerations),
        'max_acceleration': np.max(accelerations),
        'average_jerk': np.mean(jerks),
        'std_dev_jerk': np.std(jerks),
        'min_jerk': np.min(jerks),
        'max_jerk': np.max(jerks),
        'average_angular_velocity': np.mean(angular_velocities),
        'std_dev_angular_velocity': np.std(angular_velocities),
        'min_angular_velocity': np.min(angular_velocities),
        'max_angular_velocity': np.max(angular_velocities),
        'average_curvature': np.mean(curvatures),
        'std_curvature': np.std(curvatures),
        'min_curvature': np.min(curvatures),
        'max_curvature': np.max(curvatures),
        'total_time': total_time,
        'trajectory_length': trajectory_length,
        'straight_distance': straight_distance,
        'straightness': straightness,
        'direction': direction,
        'point_count': point_count,
        'total_angle': total_angle,
        'initial_acceleration': initial_acceleration
    }
    return features

def extract_features_from_dir(ds_dir, attach_filename=False):
    """
    Extract features from all files in a given directory.

    For each file:
    - Loads and cleans data.
    - Segments into actions.
    - Computes features for each action.

    Optionally appends filename and file_id columns.

    Parameters
    ----------
    ds_dir : str
        Directory path containing mouse data files.
    attach_filename : bool, optional
        If True, include '__filename' and 'file_id' columns.

    Returns
    -------
    pandas.DataFrame
        Combined DataFrame of features for all actions in all files.
    """

    all_sections = []
    file_counter = 0

    for filename in sorted(os.listdir(ds_dir)):
        if not filename.startswith("session_"):
            continue
                
        file_path = os.path.join(ds_dir, filename)

        if not os.path.isfile(file_path):
            continue

        cleaned = clean_mouse_data(file_path)
        segmented = segment_mouse_actions(cleaned)

        rows = []
        for action in segmented:
            feats = compute_features(action[0])
            if feats is None:
                continue
            row = dict(feats)
            row["action_type"] = action[1]
            if attach_filename:
                row["__filename"] = filename
                row["file_id"] = file_counter
            rows.append(row)

        if rows:
            df_feats = pd.DataFrame(rows)
            all_sections.append(df_feats)
            if attach_filename:
                file_counter += 1

    if not all_sections:
        return pd.DataFrame()  

    return pd.concat(all_sections, ignore_index=True)

def ensure_ohe(df_action_type, ohe_cols):
    """
    One-hot encode action_type column and ensure consistent column set.

    Parameters
    ----------
    df_action_type : pandas.Series
        Series containing action type labels.
    ohe_cols : list of str
        List of expected one-hot encoded column names.

    Returns
    -------
    pandas.DataFrame
        One-hot encoded DataFrame with fixed columns (missing filled with 0).
    """

    ohe = pd.get_dummies(df_action_type, prefix='action_type')
    return ohe.reindex(columns=ohe_cols, fill_value=0)


def split_numeric(df, extra_drop_cols=None):
    """
    Select numeric columns by dropping specified ones.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    extra_drop_cols : list of str, optional
        Additional columns to drop besides 'action_type'.

    Returns
    -------
    pandas.DataFrame
        DataFrame with only numeric feature columns.
    """

    drop_cols = ['action_type']
    if extra_drop_cols:
        drop_cols += list(extra_drop_cols)
    drop_cols = [c for c in drop_cols if c in df.columns]
    return df.drop(columns=drop_cols)