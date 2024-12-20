import numpy as np
import requests
import pandas as pd
from nhl_data_processor import NHLDataProcessor

import time


class GameClient:
    def __init__(self, game_id: str, prediction_url: str, event_tracker_file: str = "event_tracker.csv"):
        self.game_id = game_id
        self.prediction_url = prediction_url
        self.tracker_file = event_tracker_file
        self.processor = NHLDataProcessor()
        self.handled_events = self._load_event_tracker()
        self.home_team_name = "Unknown Home Team"
        self.away_team_name = "Unknown Away Team"

        self.error_logs = [] #Logs for errors

    def clear_logs(self):
        self.error_logs = []

    def get_logs(self):
        return self.error_logs

    def _load_event_tracker(self) -> set:
        """Load already processed events from the tracker file."""
        try:
            df = pd.read_csv(self.tracker_file)
            return set(df['event_id'])
        except FileNotFoundError:
            return set()

    def _update_event_tracker(self, new_event_ids: list):
        """Update the tracker file with new event IDs."""
        df = pd.DataFrame({"event_id": list(new_event_ids)})
        df.to_csv(self.tracker_file, index=False)

    def fetch_live_game_data(self) -> dict:
        """Fetch live game data from the NHL API."""
        url = f"https://api-web.nhle.com/v1/gamecenter/{self.game_id}/play-by-play"
        try:
            response = requests.get(url)
            response.raise_for_status()
            game_data = response.json()
            # Update team names
            self.home_team_name = game_data.get("homeTeam", {}).get("commonName", {}).get("default", "Unknown Home Team")
            self.away_team_name = game_data.get("awayTeam", {}).get("commonName", {}).get("default", "Unknown Away Team")
            return game_data
        except Exception as e:
            print(f"Failed to fetch live game data: {e}")
            self.error_logs.append(f"Failed to fetch live game data: {e}")
            return {}

    def filter_new_events(self, game_data: dict) -> pd.DataFrame:
        """Filter new events that haven't been processed yet."""
        df = self.processor.dictionary_to_dataframe_single_game(game_data)
        new_events_df = df[~df['event_id'].isin(self.handled_events)]
        return new_events_df

    def send_for_prediction(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """Send new events to the prediction server and retrieve predictions."""
        try:
            # Only send relevant features for prediction
            payload = events_df[['shooting_distance', 'shot_angle']].to_dict(orient='records')
            response = requests.post(self.prediction_url, json=payload)
            response.raise_for_status()
            predictions = response.json().get("predictions", [])
            events_df["predictions"] = predictions
            return events_df
        except Exception as excp1:
            # try with only distance
            try:
                # Only send relevant features for prediction
                payload = events_df[['shooting_distance',]].to_dict(orient='records')
                response = requests.post(self.prediction_url, json=payload)
                response.raise_for_status()
                predictions = response.json().get("predictions", [])
                events_df["predictions"] = predictions
                return events_df
            except Exception as excp2:
                # try with only angle
                try:
                    # Only send relevant features for prediction
                    payload = events_df[['shot_angle',]].to_dict(orient='records')
                    response = requests.post(self.prediction_url, json=payload)
                    response.raise_for_status()
                    predictions = response.json().get("predictions", [])
                    events_df["predictions"] = predictions
                    return events_df
                except Exception as excpFinal:
                    print(f"Failed to send events for prediction: {excpFinal}")
                    self.error_logs.append(f"Failed to send events for prediction: {excpFinal}")
                    return pd.DataFrame()

    def process_game(self) -> pd.DataFrame:
        """Fetch, process, and predict data for the game."""
        game_data = self.fetch_live_game_data()
        if not game_data:
            print("No game data available.")
            self.error_logs.append(f"No game data available.")
            return pd.DataFrame()

        new_events_df = self.filter_new_events(game_data)

        # Clean the DataFrame to remove invalid values
        new_events_df = new_events_df.replace([float("inf"), -float("inf")], 0).fillna(0)
        if not new_events_df.empty:
            new_event_ids = new_events_df['event_id'].tolist()
            self.handled_events.update(new_event_ids)
            self._update_event_tracker(list(self.handled_events))
            return self.send_for_prediction(new_events_df)
        else:
            print("No new events to process.")
            self.error_logs.append(f"No new events to process.")
            return pd.DataFrame()

    def calculate_team_xg(self, processed_df: pd.DataFrame) -> dict:
        """Calculate cumulative xG for each team."""
        if processed_df.empty:
            return {"home_xG": 0.0, "away_xG": 0.0}
        xg_summary = processed_df.groupby("team_type")["predictions"].sum()
        return {
            "home_xG": xg_summary.get("home", 0.0),
            "away_xG": xg_summary.get("away", 0.0),
        }


if __name__ == "__main__":
    GAME_ID = "2016020261"  # Replace with the desired game ID
    PREDICTION_URL = "http://127.0.0.1:8000/predict"
    client = GameClient(game_id=GAME_ID, prediction_url=PREDICTION_URL)

    while True:
        processed_events = client.process_game()
        print(client.home_team_name)
        if not processed_events.empty:
            team_xg = client.calculate_team_xg(processed_events)
            print(f"Cumulative xG:\nHome Team: {team_xg['home_xG']}\nAway Team: {team_xg['away_xG']}")
        time.sleep(30)
