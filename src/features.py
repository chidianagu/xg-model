"""
Feature engineering functions for the xG model.
Takes a StatsBomb shots dataframe and adds geometric, freeze-frame-derived,
and event-context features.
"""

import numpy as np


def angle_to_goal(x, y):
    """
    Calculate the angle (in degrees) subtended by the goal mouth
    as seen from a shot location (x, y), using StatsBomb pitch coordinates.
    Goal posts are at (120, 36) and (120, 44).
    """
    post1 = np.array([120, 36])
    post2 = np.array([120, 44])
    shot = np.array([x, y])

    v1 = post1 - shot
    v2 = post2 - shot

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0

    cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angle_rad = np.arccos(cos_angle)
    return np.degrees(angle_rad)


def sign(p1, p2, p3):
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])


def point_in_triangle(p, a, b, c):
    """Return True if point p lies inside the triangle defined by a, b, c."""
    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)


def count_defenders_in_triangle(shot_location, freeze_frame):
    """
    Count opposition players positioned inside the triangle formed by
    the shot location and the two goalposts.
    """
    if not isinstance(freeze_frame, list):
        return np.nan

    post1 = (120, 36)
    post2 = (120, 44)

    count = 0
    for player in freeze_frame:
        if not player['teammate']:
            loc = player['location']
            if point_in_triangle(loc, shot_location, post1, post2):
                count += 1

    return count


def distance_to_nearest_defender(shot_location, freeze_frame):
    """Distance from the shooter to the nearest opposition player."""
    if not isinstance(freeze_frame, list):
        return np.nan

    opponent_locs = [p['location'] for p in freeze_frame if not p['teammate']]
    if not opponent_locs:
        return np.nan

    shot = np.array(shot_location)
    dists = [np.linalg.norm(shot - np.array(loc)) for loc in opponent_locs]
    return min(dists)


def goalkeeper_features(freeze_frame):
    """
    Returns (gk_distance_to_goal_line, gk_y_offset_from_center) for the
    opposition goalkeeper, or (nan, nan) if not identifiable in the frame.
    """
    if not isinstance(freeze_frame, list):
        return np.nan, np.nan

    gk = next(
        (p for p in freeze_frame
         if not p['teammate'] and p.get('position', {}).get('name') == 'Goalkeeper'),
        None
    )

    if gk is None:
        return np.nan, np.nan

    gk_x, gk_y = gk['location']
    dist_to_line = 120 - gk_x
    y_offset = abs(gk_y - 40)

    return dist_to_line, y_offset


# Named subsets used during exploration
shot_filters = {
    'headers':     lambda df: df[df['shot_body_part'] == 'Head'],
    'left_foot':   lambda df: df[df['shot_body_part'] == 'Left Foot'],
    'right_foot':  lambda df: df[df['shot_body_part'] == 'Right Foot'],
    'penalties':   lambda df: df[df['shot_type'] == 'Penalty'],
    'open_play':   lambda df: df[df['shot_type'] == 'Open Play'],
    'free_kicks':  lambda df: df[df['shot_type'] == 'Free Kick'],
    'long_shots':  lambda df: df[df['distance_to_goal'] > 20],
    'inside_box':  lambda df: df[(df['x'] >= 102) & (df['y'].between(18, 62))],
    'goals':       lambda df: df[df['shot_outcome'] == 'Goal'],
    'saved':       lambda df: df[df['shot_outcome'] == 'Saved'],
    'high_xg':     lambda df: df[df['shot_statsbomb_xg'] > 0.3],
    'low_xg':      lambda df: df[df['shot_statsbomb_xg'] < 0.05],
}


def add_all_features(shots):
    """
    Apply every feature function to a shots dataframe and return the
    augmented copy. Assumes shots has already had shootouts excluded
    and has 'location' and 'shot_freeze_frame' columns.
    """
    shots = shots.copy()

    shots['x'] = shots['location'].apply(lambda l: l[0])
    shots['y'] = shots['location'].apply(lambda l: l[1])
    shots['distance_to_goal'] = np.sqrt((120 - shots['x'])**2 + (40 - shots['y'])**2)
    shots['angle_to_goal'] = shots.apply(lambda r: angle_to_goal(r['x'], r['y']), axis=1)

    shots['defenders_in_triangle'] = shots.apply(
        lambda r: count_defenders_in_triangle(r['location'], r['shot_freeze_frame']), axis=1
    )
    shots['dist_to_nearest_defender'] = shots.apply(
        lambda r: distance_to_nearest_defender(r['location'], r['shot_freeze_frame']), axis=1
    )

    gk_stats = shots['shot_freeze_frame'].apply(goalkeeper_features)
    shots['gk_dist_to_line'] = gk_stats.apply(lambda t: t[0])
    shots['gk_y_offset'] = gk_stats.apply(lambda t: t[1])

    shots['under_pressure'] = shots['under_pressure'].fillna(False).astype(bool)
    shots['shot_open_goal'] = shots['shot_open_goal'].fillna(False).astype(bool)
    shots['shot_first_time'] = shots['shot_first_time'].fillna(False).astype(bool)
    shots['shot_one_on_one'] = shots['shot_one_on_one'].fillna(False).astype(bool)
    shots['is_counter_attack'] = shots['play_pattern'] == 'From Counter'

    return shots


MODEL_FEATURES = [
    'distance_to_goal', 'angle_to_goal',
    'defenders_in_triangle', 'dist_to_nearest_defender',
    'gk_dist_to_line', 'gk_y_offset',
    'under_pressure', 'shot_open_goal', 'shot_first_time',
    'is_counter_attack', 'shot_one_on_one'
]