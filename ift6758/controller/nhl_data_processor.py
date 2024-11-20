from nhl_data_processor_utils import parse_situation_code, get_strength_status, is_net_empty_goal, get_home_team_defending_side, calculate_duration_mm_ss, euclidean_distance, calculate_shooting_distance, compute_angle, compute_angle_row
from typing import List
import os
import json
import numpy as np
import pandas as pd

class NHLDataProcessor():
    def __init__(self):
        self.data_dir_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        os.makedirs(os.path.join(self.data_dir_path, "play_by_play", "csv"), exist_ok=True)

    def dictionary_to_dataframe_single_game(self, game_data):
        events = game_data.get("plays", [])
        extracted_events = []

        # Get home and away team information
        home_team_id = game_data.get("homeTeam", {}).get("id", None)
        away_team_id = game_data.get("awayTeam", {}).get("id", None)
        home_team_name = game_data.get("homeTeam", {}).get("name", {}).get("default", None)
        away_team_name = game_data.get("awayTeam", {}).get("name", {}).get("default", None)

        for event in events:
            details = event.get("details", {})
            event_type = event.get("typeDescKey", "")
            situation_code = event.get("situationCode", None)
            parsed_situation = parse_situation_code(situation_code) if situation_code else {}
            strength_status = get_strength_status(parsed_situation)

            # Get the number of skaters for both teams
            home_skaters = parsed_situation.get('home_skaters', 0)
            away_skaters = parsed_situation.get('away_skaters', 0)

            # Add the real strength info as 'XvY' (e.g., 5v4)
            real_strength = f"{home_skaters}v{away_skaters}"

            # Determine if the team is home or away
            event_owner_team_id = details.get("eventOwnerTeamId", None)
            if event_owner_team_id == home_team_id:
                team_type = "home"
                team_name = home_team_name
            elif event_owner_team_id == away_team_id:
                team_type = "away"
                team_name = away_team_name
            else:
                team_type = None
                team_name = None

            # Determine if the net is empty for the current team
            empty_net_status = is_net_empty_goal(team_type, parsed_situation)

            # Default shot type handling
            shot_type = details.get("shotType", None)

            x_coord = details.get("xCoord", None)
            y_coord = details.get("yCoord", None)

            # Assign shooter_id based on event type
            shooter_id = (
                details.get("scoringPlayerId", None)
                if event_type == "goal"
                else details.get("shootingPlayerId", None)
            )

            # Collect event information
            event_info = {
                "game_id": game_data.get("id", None),
                "game_date": game_data.get("gameDate", None),
                "home_team_id": home_team_id,
                "period": event.get("periodDescriptor", {}).get("number", None),
                "time_in_period": event.get("timeInPeriod", None),
                "event_id": event.get("eventId", None),
                "event_type": event_type,
                "is_goal": event_type == "goal",
                "shot_type": shot_type,
                "x_coord": x_coord,
                "y_coord": y_coord,
                "event_owner_team_id": details.get("eventOwnerTeamId", None),
                "team_name": team_name,
                "team_type": team_type,
                "empty_net": empty_net_status,
                "strength_status": strength_status,
                "real_strength_home_vs_away": real_strength,
                "situation_code": situation_code,
                "shooter_id": shooter_id,
                "goalie_id": details.get("goalieInNetId", None),
            }
            extracted_events.append(event_info)

        # Convert extracted events to a DataFrame
        df = pd.DataFrame(extracted_events)
        return df

    def sort(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.sort_values(by=["game_id", "period", "time_in_period"], ascending=True)
        df.reset_index(drop=True, inplace=True)
        return df
    
    def add_last_event_type_column(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['last_event_type'] = 'unknown'
        mask_shot_events = df['event_type'].isin(['shot-on-goal', 'goal'])
        df['prev_game_id'] = df['game_id'].shift(1)
        df['prev_period'] = df['period'].shift(1)
        df['prev_event_type'] = df['event_type'].shift(1)
        same_game_and_period = (df['game_id'] == df['prev_game_id']) & (df['period'] == df['prev_period'])
        df.loc[mask_shot_events & same_game_and_period, 'last_event_type'] = df.loc[
            mask_shot_events & same_game_and_period, 'prev_event_type'
        ]
        df.drop(['prev_game_id', 'prev_period', 'prev_event_type'], axis=1, inplace=True)
        return df
    
    def add_last_x_coord_column(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['last_x_coord'] = 'unknown'
        mask_shot_events = df['event_type'].isin(['shot-on-goal', 'goal'])
        df['prev_game_id'] = df['game_id'].shift(1)
        df['prev_period'] = df['period'].shift(1)
        df['prev_x_coord'] = df['x_coord'].shift(1)
        same_game_and_period = (df['game_id'] == df['prev_game_id']) & (df['period'] == df['prev_period'])
        df.loc[mask_shot_events & same_game_and_period, 'last_x_coord'] = df.loc[
            mask_shot_events & same_game_and_period, 'prev_x_coord'
        ]
        df.drop(['prev_game_id', 'prev_period', 'prev_x_coord'], axis=1, inplace=True)
        return df
    
    def add_last_y_coord_column(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['last_y_coord'] = 'unknown'
        mask_shot_events = df['event_type'].isin(['shot-on-goal', 'goal'])
        df['prev_game_id'] = df['game_id'].shift(1)
        df['prev_period'] = df['period'].shift(1)
        df['prev_y_coord'] = df['y_coord'].shift(1)
        same_game_and_period = (df['game_id'] == df['prev_game_id']) & (df['period'] == df['prev_period'])
        df.loc[mask_shot_events & same_game_and_period, 'last_y_coord'] = df.loc[
            mask_shot_events & same_game_and_period, 'prev_y_coord'
        ]
        df.drop(['prev_game_id', 'prev_period', 'prev_y_coord'], axis=1, inplace=True)
        return df
    
    def add_time_since_last_event_column(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['time_since_last_event'] = np.nan  # Initialiser avec NaN pour une durée
        mask_shot_events = df['event_type'].isin(['shot-on-goal', 'goal'])
        
        # Colonnes temporaires pour décalage
        df['prev_game_id'] = df['game_id'].shift(1)
        df['prev_period'] = df['period'].shift(1)
        df['prev_time_in_period'] = df['time_in_period'].shift(1)
        
        # Vérifier si le jeu et la période sont identiques
        same_game_and_period = (df['game_id'] == df['prev_game_id']) & (df['period'] == df['prev_period'])
        
        # Calculer les durées avec `apply`
        df.loc[mask_shot_events & same_game_and_period, 'time_since_last_event'] = df.loc[mask_shot_events & same_game_and_period].apply(
            lambda row: calculate_duration_mm_ss(row["prev_time_in_period"], row["time_in_period"]),
            axis=1
        )
        
        # Supprimer les colonnes temporaires
        df.drop(['prev_game_id', 'prev_period', 'prev_time_in_period'], axis=1, inplace=True)
        
        return df
    
    def add_distance_from_last_event_column(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['distance_from_last_event'] = np.nan
        mask_shot_events = df['event_type'].isin(['shot-on-goal', 'goal'])
        df['prev_game_id'] = df['game_id'].shift(1)
        df['prev_period'] = df['period'].shift(1)
        same_game_and_period = (df['game_id'] == df['prev_game_id']) & (df['period'] == df['prev_period'])
        df.loc[mask_shot_events & same_game_and_period, 'distance_from_last_event'] = df.loc[mask_shot_events & same_game_and_period].apply(
            lambda row: euclidean_distance(row['last_x_coord'], row['last_y_coord'], row['x_coord'], row['y_coord']),
            axis=1
        )
        df.drop(['prev_game_id', 'prev_period'], axis=1, inplace=True)
        return df
    
    def add_rebound_column(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['rebound'] = False
        df['prev_event_type'] = df['event_type'].shift(1)
        mask_shot_events = df['event_type'].isin(['shot-on-goal', 'goal'])
        df.loc[mask_shot_events, 'rebound'] = df.loc[mask_shot_events, 'prev_event_type'].isin(['shot-on-goal', 'goal'])
        df.drop(['prev_event_type'], axis=1, inplace=True)
        return df
    
    def add_home_team_defending_side_column(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df['home_team_defending_side'] = None
        number_of_lines = df.shape[0]
        games_ids = []
        home_team_defending_side_first_period = {}
        home_team_defending_side_second_period = {}

        for index in range(number_of_lines):
            game_id = df.loc[index, 'game_id']
            if game_id in games_ids:
                period = df.loc[index, 'period']
                if period % 2 == 1:
                    df.loc[index, 'home_team_defending_side'] = home_team_defending_side_first_period[game_id]
                else:
                    df.loc[index, 'home_team_defending_side'] = home_team_defending_side_second_period[game_id]
            else:
                period = df.loc[index, 'period']
                filtered_df = df.query(f"game_id == {game_id} and period == {period} and team_type == 'home'")
                x_coords = filtered_df['x_coord']
                median = x_coords.median()
                if median > 0:
                    df.loc[index, 'home_team_defending_side'] = 'left'
                    if period % 2 == 1:
                        home_team_defending_side_first_period[game_id] = 'left'
                        home_team_defending_side_second_period[game_id] = 'right'
                    else:
                        home_team_defending_side_first_period[game_id] = 'right'
                        home_team_defending_side_second_period[game_id] = 'left'
                else:
                    df.loc[index, 'home_team_defending_side'] = 'right'
                    if period % 2 == 1:
                        home_team_defending_side_first_period[game_id] = 'right'
                        home_team_defending_side_second_period[game_id] = 'left'
                    else:
                        home_team_defending_side_first_period[game_id] = 'left'
                        home_team_defending_side_second_period[game_id] = 'right'
                games_ids.append(game_id)
                
        return df
    
    def add_shooting_distance_column(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['shooting_distance'] = np.nan
        mask_shot_events = df['event_type'].isin(['shot-on-goal', 'goal'])
        df.loc[mask_shot_events, 'shooting_distance'] = df.loc[mask_shot_events].apply(calculate_shooting_distance, axis=1)
        return df
    
    def add_shot_angle_column(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['shot_angle'] = np.nan
        mask_shot_events = df['event_type'].isin(['shot-on-goal', 'goal'])
        df.loc[mask_shot_events, 'shot_angle'] = df.loc[mask_shot_events].apply(compute_angle_row, axis=1)
        return df
    
    def add_rebound_angle_column(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['rebound_angle'] = np.nan
        df['prev_shot_angle'] = df['shot_angle'].shift(1)
        mask_rebound_events = df['rebound'] == True
        df.loc[mask_rebound_events, 'rebound_angle'] = np.abs(df.loc[mask_rebound_events, 'shot_angle'] - df.loc[mask_rebound_events, 'prev_shot_angle'])
        df.drop(['prev_shot_angle'], axis=1, inplace=True)
        return df
    
    def add_speed_from_last_event_column(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['speed_from_last_event'] = np.nan
        mask_shot_events = df['event_type'].isin(['shot-on-goal', 'goal'])
        df.loc[mask_shot_events, 'speed_from_last_event'] = df.loc[mask_shot_events, 'distance_from_last_event'] / df.loc[mask_shot_events, 'time_since_last_event']
        return df
