import os
import pandas as pd
import requests
import json

from IPython.display import display
import ipywidgets as widgets

class NHLDataDownloader:
    DATA_DIR = "../data/play_by_play"
    BASE_URL = "https://api-web.nhle.com/v1/gamecenter/"
    PLAY_BY_PLAY_SUFFIX_URL = "/play-by-play"

    def __init__(self, base_url="https://api-web.nhle.com/v1/gamecenter/", data_dir="../data/play_by_play", suffix_url="/play-by-play"):
        self.base_url = base_url
        # Ensure the play_by_play folder is always under ift6758/data
        self.data_dir = os.path.join(os.path.dirname(__file__), data_dir)
        self.suffix_url = suffix_url
        os.makedirs(data_dir, exist_ok=True)

        self.progress_bar = widgets.FloatProgress(
                value=0.0,
                min=0.0,
                max=1.0,
                description='Loading:',
            )

    def save_and_get_game_data(self, game_id, season):
        """Download data for a specific game ID"""
        game_file = f"{self.data_dir}/{season}/{game_id}.json"
        os.makedirs(f"{self.data_dir}/{season}", exist_ok=True)

        if os.path.exists(game_file):
            print(f"Game data for {game_id} already exists in local cache.")
            with open(game_file, "r") as file:
                return json.load(file)

        url = f"{self.base_url}{game_id}{self.suffix_url}"
        response = requests.get(url)

        if response.status_code == 200:
            try:
                data = json.loads(response.text)
                with open(game_file, "w") as file:
                    json.dump(data, file)
                return data
            except json.JSONDecodeError as e:
                print(f"Failed to decode cleaned JSON: {e}")
                return None
        else:
            print(f"Failed to download data for game {game_id}.")
            return None

    @staticmethod
    def generate_regular_season_game_id(season, game_number):
        """Generate a game ID for a regular season game"""
        season_str = str(season)
        game_type_str = "02" # 02 is the game type for regular season games
        game_number_str = f"{game_number:04d}"
        return f"{season_str}{game_type_str}{game_number_str}"

    @staticmethod
    def generate_playoff_game_id(season, round_num,matchup, game_num):
        """
        Generate a game ID for a playoff game.
        Args:
            - season: The starting year of the season (e.g., 2023 for 2023-2024 season)
            - round_num: The round of the playoffs (01 for first round, 02 for second round, etc.max 04)
            - matchup: The matchup number within that round
            - game_num: The game number in the series (1 to 7)
        """
        season_str = str(season)
        game_type_str = "03"
        game_number_str = f"{round_num:02d}{matchup}{game_num}"
        return f"{season_str}{game_type_str}{game_number_str}"

    def download_regular_season(self, season, total_games=1353, output_widget=None):
        """Download all regular season games for a given season"""

        self.progress_bar.value = 0
        if output_widget == None:
            display(self.progress_bar)
        else:
            with output_widget:
                display(self.progress_bar)

        os.makedirs(f"{self.data_dir}/{season}_CleanCSV", exist_ok=True)

        all_games = {}
        for game_num in range(1, total_games+1):
            game_id = NHLDataDownloader.generate_regular_season_game_id(season, game_num)
            game_data = self.save_and_get_game_data(game_id, season)
            if game_data:
                all_games[game_id] = game_data
                # Save the cleaned dataframe version
                clean_game_file = f"{self.data_dir}/{season}_CleanCSV/{game_id}.csv"
                clean_df = self.extract_shots_and_goals(game_data, season)
                clean_df.to_csv(clean_game_file,index=False)

            self.progress_bar.value = (game_num-1) / total_games

        return all_games

    def download_playoff_series(self, season, round_num, matchup, total_games=7):
        """Download all games in a playoff series"""
        series_data = {}
        for game_num in range(1, total_games+1):
            game_id = self.generate_playoff_game_id(season, round_num, matchup, game_num)
            game_data = self.save_and_get_game_data(game_id, season)
            if game_data:
                series_data[game_id] = game_data
                # Save the cleaned dataframe version
                clean_game_file = f"{self.data_dir}/{season}_CleanCSV/{game_id}.csv"
                clean_df = self.extract_shots_and_goals(game_data, season)
                clean_df.to_csv(clean_game_file,index=False)

        return series_data

    def extract_shots_and_goals_for_game(self, game_id, season):
        """Extract shot and goal data for a specific game"""
        game_data = self.save_and_get_game_data(game_id, season)
        if game_data:
            return self.extract_shots_and_goals(game_data, season)
        else:
            print(f"Failed to extract shot and goal data for game {game_id}.")
            return None

    def download_playoffs(self, season, output_widget=None):
        """
        Download data for the entire playoffs, dynamically adjusting matchups for each round:
            - Round 1: 8 matchups
            - Round 2: 4 matchups
            - Round 3: 2 matchups
            - Round 4: 1 matchup (Stanley Cup Final)
        """
        self.progress_bar.value = 0.0
        if output_widget == None:
            display(self.progress_bar)
        else:
            with output_widget:
                display(self.progress_bar)

        rounds_matchups = {
            1: 8,  # Round 1 has 8 matchups
            2: 4,  # Round 2 has 4 matchups
            3: 2,  # Round 3 has 2 matchups
            4: 1  # Round 4 (Stanley Cup Final) has 1 matchup
        }

        os.makedirs(f"{self.data_dir}/{season}_CleanCSV", exist_ok=True)

        all_games = {}
        for round_num, matchups in rounds_matchups.items():
            for matchup in range(1, matchups + 1):
                # Each matchup can have up to 7 games in a best-of-seven series
                game_data = self.download_playoff_series(season, round_num, matchup)
                if game_data:
                    all_games.update(game_data)
                self.progress_bar.value += (1/matchups) / matchups
        return all_games

    def download_all_seasons_play_by_play(self, start_season, end_season):
        """Download all regular season games for a range of seasons"""
        for season in range(start_season, end_season+1):
            self.download_regular_season(season)
            self.download_playoffs(season)

    ########################################################################################
    # Method below are used for data cleaning
    ########################################################################################

    #Helper function to parse the situation code
    def parse_situation_code(self, situation_code):
        """
        Parse the situation code and return a dictionary with the parsed values.
        1st: away goalie (1=in net, 0=pulled)
        2nd: number of away skaters
        3rd: number of home skaters
        4th: home goalie (1=in net, 0=pulled)
        """
        if len(situation_code) == 4:
            away_goalie_in_net = situation_code[0] == '1'
            away_skaters = int(situation_code[1])
            home_skaters = int(situation_code[2])
            home_goalie_in_net = situation_code[3] == '1'
            return {
                'away_goalie_in_net': away_goalie_in_net,
                'away_skaters': away_skaters,
                'home_skaters': home_skaters,
                'home_goalie_in_net': home_goalie_in_net
            }

    #Helper function to get the strength status
    def get_strength_status(self, parsed_situation):
        """
        Determine whether the goal or shot occurred during equal strength, power play, or penalty kill.

        Arguments:
        - parsed_situation: Dictionary containing details of skaters and goalie presence.

        Returns:
        - A string indicating 'Equal Strength', 'Power Play', or 'Penalty Kill'
        """
        away_skaters = parsed_situation.get('away_skaters', 0)
        home_skaters = parsed_situation.get('home_skaters', 0)

        if home_skaters == away_skaters:
            return "Equal Strength"
        elif home_skaters > away_skaters:
            return "Power Play"  # Home team has more skaters
        else:
            return "Penalty Kill"  # Home team has fewer skaters

    #Helper function to determine if the net is empty
    def is_net_empty_goal(self, team_type, parsed_situation):
        """
        Determine if the net is empty based on the team type (home or away).

        Args:
        - team_type: A string indicating whether the team is 'home' or 'away'
        - parsed_situation: A dictionary containing parsed situation details (goalie in net, skaters, etc.)

        Returns:
        - True if the net is empty for the given team, False otherwise.
        """
        if team_type == "home":
            return not parsed_situation.get('away_goalie_in_net', True)  # If away goalie is not in net
        elif team_type == "away":
            return not parsed_situation.get('home_goalie_in_net', True)  # If home goalie is not in net
        return False  # Default: net is not empty if team type is unknown

    #Helper function to determine the home team's defending side
    @staticmethod
    def get_home_team_defending_side(x_coord, event_owner_team_id, home_team_id, zone_code, previous_defending_side):
        """
        Determines which side ("left" or "right") is the home team’s defensive zone
        based on the x-coordinate, event owner, and zone code.

        Parameters:
        - x_coord: x-coordinate of the event.
        - event_owner_team_id: ID of the team associated with the event.
        - home_team_id: The ID of the home team.
        - zone_code: Zone code indicating if the event is offensive ("O") or defensive ("D").

        Returns:
        - "left" or "right" indicating the defensive side of the home team for the event.
        """
        # Case 1: Event involves the home team
        if event_owner_team_id == home_team_id:
            if zone_code == "O":  # Offensive zone for home team
                # Current side is offensive, so the opposite side is defensive
                if x_coord > 0:
                    return "left"  # Right is offensive, so left is defensive
                elif x_coord < 0:
                    return "right"  # Left is offensive, so right is defensive
            elif zone_code == "D":  # Defensive zone for home team
                # Current side is defensive
                if x_coord > 0:
                    return "right"  # Right is defensive
                elif x_coord < 0:
                    return "left"  # Left is defensive

        # Case 2: Event involves the away team
        else:
            if zone_code == "D":  # Defensive zone for home team
                # Current side is defensive
                if x_coord > 0:
                    return "left"  # Right is defensive for away, so left is defensive for home
                elif x_coord < 0:
                    return "right"  # Left is defensive for away, so right is defensive for home
            elif zone_code == "O":  # Offensive zone for away team
                # Current side is offensive
                if x_coord > 0:
                    return "right"  # Right is offensive for away, so right is defensive for home
                elif x_coord < 0:
                    return "left"  # Left is offensive for away, so left is defensive for home
        if zone_code == "N":
            return previous_defending_side
        return None

    #Helper function to extract shots and goals
    def extract_shots_and_goals(self, game_data, season):
        """Extract 'shots-on-goal' and 'hit' from the game data and return them as a pandas DataFrame."""
        events = game_data.get("plays",[])
        extracted_events = []

        # Get home and away team information
        home_team_id = game_data.get("homeTeam", {}).get("id", None)
        away_team_id = game_data.get("awayTeam", {}).get("id", None)
        home_team_name = game_data.get("homeTeam", {}).get("name", {}).get("default", "Unknown")
        away_team_name = game_data.get("awayTeam", {}).get("name", {}).get("default", "Unknown")

        previous_defending_side = None
        for event in events:
            details = event.get("details",{})
            event_type =event.get("typeDescKey","")
            situation_code = event.get("situationCode",None)

            if event_type in ["shot-on-goal","goal"]:
                parsed_situation = self.parse_situation_code(situation_code) if situation_code else {}
                strength_status = self.get_strength_status(parsed_situation)

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
                    team_type = "unknown"
                    team_name = "unknown"

                # Determine if the net is empty for the current team
                empty_net_status = self.is_net_empty_goal(team_type, parsed_situation)
                # Default shot type handling
                shot_type = details.get("shotType", "Unknown")

                # Determine the home team's defending side for this specific event
                x_coord = details.get("xCoord", None)
                if season < 2019:
                    zone_code = details.get("zoneCode", "")
                    home_team_defending_side = self.get_home_team_defending_side(x_coord, event_owner_team_id, home_team_id, zone_code, previous_defending_side)
                    if home_team_defending_side is not None:
                        previous_defending_side = home_team_defending_side
                else:
                    home_team_defending_side = event.get("homeTeamDefendingSide", None)

                event_info ={
                    "game_id":game_data.get("id",None),
                    "game_date":game_data.get("gameDate",None),
                    "period": event.get("periodDescriptor",{}).get("number",None),
                    "time_in_period": event.get("timeInPeriod",None),
                    "event_id": event.get("eventId",None),
                    "event_type": event_type,
                    "is_goal": event_type == "goal",
                    "shot_type": shot_type,
                    "x_coord": x_coord,
                    "y_coord": details.get("yCoord", None),
                    "event_owner_team_id": details.get("eventOwnerTeamId",None),
                    "home_team_defending_side": home_team_defending_side,
                    "team_name": team_name,
                    "team_type": team_type,
                    "empty_net": empty_net_status,
                    "strength_status": strength_status,
                    "real_strength_home_vs_away": real_strength,
                    "situation_code": situation_code,
                    "shooter_id": details.get("shootingPlayerId", "Unknown"),
                    "goalie_id": details.get("goalieInNetId", "Unknown"),
                }
                extracted_events.append(event_info)

        df = pd.DataFrame(extracted_events)

        return df



#def get_game_dataframe(game_id):
#    season = str(game_id[:4])
#    clean_game_file = f"{NHLDataDownloader.DATA_DIR}/{season}_CleanCSV/{game_id}.csv"
#    os.makedirs(f"{NHLDataDownloader.DATA_DIR}/{season}_CleanCSV", exist_ok=True)

#    if os.path.exists(clean_game_file):
#        return pd.read_csv(clean_game_file)
#    else:
#        url = f"{NHLDataDownloader.BASE_URL}{game_id}{NHLDataDownloader.PLAY_BY_PLAY_SUFFIX_URL}"
#        response = requests.get(url)
#        if response.status_code == 200:
#            try:
#                game_data = json.loads(response.text)

#                downloader = NHLDataDownloader()
#                # Save the cleaned dataframe version
#                clean_game_file = f"{NHLDataDownloader.DATA_DIR}/{season}_CleanCSV/{game_id}.csv"
#                clean_df = downloader.extract_shots_and_goals(game_data)
#                clean_df.to_csv(clean_game_file)
#                return clean_df

#            except json.JSONDecodeError as e:
#                print(f"Failed to decode cleaned JSON: {e}")
#                return None
#        else:
#            print(f"Failed to download data for game {game_id}.")
#            return None



def get_dataframe_from_csv_file(game_id: str) -> pd.DataFrame:
    season = str(game_id[:4])
    csv_file_path_name = f"{NHLDataDownloader.DATA_DIR}/{season}_CleanCSV/{game_id}.csv"

    if os.path.exists(csv_file_path_name):
        return pd.read_csv(csv_file_path_name)
    else:
        print(f"Le fichier {game_id}.csv n'existe pas.")
        return None



#def clean_games_data(season: int):
#    season_games_data_path_name = f"{NHLDataDownloader.DATA_DIR}/{season}/"
#    json_files_names = [json_file_name for json_file_name in os.listdir(season_games_data_path_name) if json_file_name.endswith('.json')]

#    for json_file_name in json_files_names:
#        game_id = json_file_name.replace('.json', '')
#        get_game_dataframe(game_id)



def get_dataframe_from_concatenated_csv_files(season: int) -> pd.DataFrame:
    season_games_cleaned_data_path_name = f"{NHLDataDownloader.DATA_DIR}/{season}_CleanCSV/"
    csv_files_names = [csv_file_name for csv_file_name in os.listdir(season_games_cleaned_data_path_name) if csv_file_name.endswith('.csv')]
    all_df = []

    for csv_file_name in csv_files_names:
        csv_file_path_name = os.path.join(season_games_cleaned_data_path_name, csv_file_name)
        df = pd.read_csv(csv_file_path_name)
        all_df.append(df)
    combined_df = pd.concat(all_df, ignore_index=True)

    return combined_df