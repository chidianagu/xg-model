"""
Data loading functions for pulling StatsBomb shot data across
one or more competitions.
"""

import pandas as pd
from statsbombpy import sb


def load_shots(competitions):
    """
    competitions: list of (competition_id, season_id) tuples.
    Returns a combined shots dataframe across all given competitions,
    with team/stage info merged in and penalty shootouts (period 5) excluded.
    """
    all_matches = []
    for comp_id, season_id in competitions:
        m = sb.matches(competition_id=comp_id, season_id=season_id)
        m['competition_id'] = comp_id
        m['season_id'] = season_id
        all_matches.append(m)
    matches = pd.concat(all_matches, ignore_index=True)

    all_shots = []
    for match_id in matches['match_id']:
        events = sb.events(match_id=match_id)
        match_shots = events[events['type'] == 'Shot'].copy()
        match_shots['match_id'] = match_id
        all_shots.append(match_shots)

    shots = pd.concat(all_shots, ignore_index=True)

    shots = shots.merge(
        matches[['match_id', 'home_team', 'away_team', 'competition_stage',
                 'competition_id', 'season_id']],
        on='match_id', how='left'
    )

    shots = shots[shots['period'] != 5].copy()  # exclude penalty shootouts

    return shots