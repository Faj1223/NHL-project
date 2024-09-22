import os
import requests
import json

class NHLDataDownloader:
    def __init__(self, base_url="https://api-web.nhle.com/v1/gamecenter/", data_dir="play_by_play", suffix_url="/play-by-play"):
        self.base_url = base_url
        # Ensure the play_by_play folder is always under ift6758/data
        self.data_dir = os.path.join(os.path.dirname(__file__), 'play_by_play')
        self.suffix_url = suffix_url
        os.makedirs(data_dir, exist_ok=True)

    def get_game_data(self, game_id):
        """Download data for a specific game ID"""
        season = str(game_id[:4])
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

    def download_regular_season(self, season, total_games=1353):
        """Download all regular season games for a given season"""
        for game_num in range(1, total_games+1):
            game_id = NHLDataDownloader.generate_regular_season_game_id(season, game_num)
            self.get_game_data(game_id)

    def download_playoff_series(self, season, round_num, matchup, total_games=7):
        """Download all games in a playoff series"""
        for game_num in range(1, total_games+1):
            game_id = self.generate_playoff_game_id(season, round_num, matchup, game_num)
            self.get_game_data(game_id)

    def download_playoffs(self, season):
        """
        Download data for the entire playoffs, dynamically adjusting matchups for each round:
            - Round 1: 8 matchups
            - Round 2: 4 matchups
            - Round 3: 2 matchups
            - Round 4: 1 matchup (Stanley Cup Final)
        """
        rounds_matchups = {
            1: 8,  # Round 1 has 8 matchups
            2: 4,  # Round 2 has 4 matchups
            3: 2,  # Round 3 has 2 matchups
            4: 1  # Round 4 (Stanley Cup Final) has 1 matchup
        }

        for round_num, matchups in rounds_matchups.items():
            for matchup in range(1, matchups + 1):
                # Each matchup can have up to 7 games in a best-of-seven series
                self.download_playoff_series(season, round_num, matchup)

    def download_all_seasons_play_by_play(self, start_season, end_season):
        """Download all regular season games for a range of seasons"""
        for season in range(start_season, end_season+1):
            self.download_regular_season(season)
            self.download_playoffs(season)




