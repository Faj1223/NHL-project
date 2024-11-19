
from ift6758.controller.nhl_data_downloader  import NHLDataDownloader, get_dataframe_from_csv_file, get_dataframe_from_concatenated_csv_files
import numpy as np
import pandas as pd
from scipy.spatial import distance

class TableFormatter:
    def __init__(self, start_year, end_year):
        self.start_year = start_year
        self.end_year = end_year

        self.whole_df = get_dataframe_from_concatenated_csv_files(start_year, end_year)
        if "Unknown" in self.whole_df.columns:
            self.whole_df = self.whole_df.drop(index=["Unknown"])

        self.formatted_df = self.whole_df.copy()

    def remove_column(self, column_name: str):
        """
        Remove specified column from the formatted table
        Args:
            column_name: string of column to remove
        """
        self.formatted_df = self.formatted_df.drop(column_name)


    def add_home_team_defending_side(self):
        """
        Add 'home_team_defending_side' column to the formatted table
        Based on the shots made by a team. If most shot were on a certain side during a period, they had
        their goal on the opposite side.
        """
        self.formatted_df['home_team_defending_side'] = None
        number_of_lines = self.formatted_df.shape[0]
        games_ids = []
        home_team_defending_side_first_period = {}
        home_team_defending_side_second_period = {}

        for index in range(number_of_lines):
            game_id =  self.formatted_df.loc[index, 'game_id']
            if game_id in games_ids:
                period = self.formatted_df.loc[index, 'period']
                if period % 2 == 1:
                    self.formatted_df.loc[index, 'home_team_defending_side'] = home_team_defending_side_first_period[game_id]
                else:
                    self.formatted_df.loc[index, 'home_team_defending_side'] = home_team_defending_side_second_period[game_id]
            else:
                period =  self.formatted_df.loc[index, 'period']
                filtered_df =  self.formatted_df.query(f"game_id == {game_id} and period == {period} and team_type == 'home'")
                x_coords = filtered_df['x_coord']
                median = x_coords.median()
                if median > 0:
                    self.formatted_df.loc[index, 'home_team_defending_side'] = 'left'
                    if period % 2 == 1:
                        home_team_defending_side_first_period[game_id] = 'left'
                        home_team_defending_side_second_period[game_id] = 'right'
                    else:
                        home_team_defending_side_first_period[game_id] = 'right'
                        home_team_defending_side_second_period[game_id] = 'left'
                else:
                    self.formatted_df.loc[index, 'home_team_defending_side'] = 'right'
                    if period % 2 == 1:
                        home_team_defending_side_first_period[game_id] = 'right'
                        home_team_defending_side_second_period[game_id] = 'left'
                    else:
                        home_team_defending_side_first_period[game_id] = 'left'
                        home_team_defending_side_second_period[game_id] = 'right'
                games_ids.append(game_id)

    def __shot_distance(self, shots: pd.Series) -> float:
        """
        Calculate distance from shot to goal
        Make sure 'home_team_defending_side' was added before this is called
        """
        shooter_coords = (shots['x_coord'], shots['y_coord'])

        if shots['home_team_defending_side'] == 'left':
            home_goalie_coords = (-100, 0)
            away_goalie_coords = (100, 0)
        else:
            home_goalie_coords = (100, 0)
            away_goalie_coords = (-100, 0)

        if shots['team_type'] == 'home':
            return float(distance.euclidean(shooter_coords, away_goalie_coords))
        else:
            return float(distance.euclidean(shooter_coords, home_goalie_coords))

    def add_shot_distance(self):
        """
        Add distance of the shot given the relative goalie.
        Make sure 'home_team_defending_side' was added before this is called
        """
        self.formatted_df['shot_distance'] = self.formatted_df.apply(self.__shot_distance, axis=1)

    def format_empty_net_as_01(self):
        self.formatted_df['empty_net'] = self.formatted_df['empty_net'].astype(int)

    def format_is_goal_as_01(self):
        self.formatted_df['is_goal'] = self.formatted_df['is_goal'].astype(int)

    def __relative_shot_angle(self, shots: pd.Series) -> float:
        """
        Calculate angle between shot to goal's norm
        Make sure 'home_team_defending_side' was added before this is called
        Returns: the angle in radians
        """
        shooter_coords = np.array( [shots['x_coord'], shots['y_coord']])

        if shots['home_team_defending_side'] == 'left':
            home_goalie_coords = np.array( [-100, 0])
            away_goalie_coords = np.array( [100, 0])
        else:
            home_goalie_coords = np.array( [100, 0])
            away_goalie_coords = np.array( [-100, 0])

        shot_vector = (shooter_coords - home_goalie_coords)

        # goal norm is the normalized vector from goal to 0,0
        if shots['team_type'] == 'home':
            goal_norm = (-1, 0) #here we suppose we always shoot toward enemy goal
        else:
            goal_norm = (1, 0)  # here we suppose we always shoot toward enemy goal

        return np.arccos(np.dot(shot_vector, goal_norm) / (np.linalg.norm(shot_vector)))

    def add_relative_shot_angle(self):
        """
        Add radian angle between shot vector and goal's norm
        Make sure 'home_team_defending_side' was added before this is called
        """
        self.formatted_df['relative_shot_angle'] = self.formatted_df.apply(self.__relative_shot_angle, axis=1)

    def get_milestone_2_q2_formatting(self):
        # Étape 1 : Supprimer les lignes avec des valeurs manquantes dans 'x_coord' et 'y_coord' (elles sont insignifiantes en terme de pourcentage)
        self.formatted_df = self.formatted_df.dropna(subset=['x_coord', 'y_coord'])
        # Renuméroter les indices de manière séquentielle
        self.formatted_df = self.formatted_df.reset_index(drop=True)

        self.add_home_team_defending_side()
        self.add_shot_distance()
        self.add_relative_shot_angle()

        self.format_is_goal_as_01()
        self.format_empty_net_as_01()


"""
Distance du filet
Angle relatif au filet
est un but (0 ou 1)
Filet vide (0 ou 1, vous pouvez supposons que les NaN sont 0)
"""