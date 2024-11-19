from .nhl_data_processor_utils import parse_situation_code, get_strength_status, is_net_empty_goal, get_home_team_defending_side, calculate_distance_and_angle
from typing import List
import os
import json
import pandas as pd

class NHLDataProcessor():
    def __init__(self):
        self.data_dir_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        os.makedirs(os.path.join(self.data_dir_path, "play_by_play", "csv"), exist_ok=True)

    def dict_to_data_frame_single_game(self, game_data):
        events = game_data.get("plays", [])
        extracted_events = []

        # Get home and away team information
        home_team_id = game_data.get("homeTeam", {}).get("id", None)
        away_team_id = game_data.get("awayTeam", {}).get("id", None)
        home_team_name = game_data.get("homeTeam", {}).get("name", {}).get("default", "Unknown")
        away_team_name = game_data.get("awayTeam", {}).get("name", {}).get("default", "Unknown")

        previous_defending_side = None

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
                team_type = "unknown"
                team_name = "unknown"

            # Determine if the net is empty for the current team
            empty_net_status = is_net_empty_goal(team_type, parsed_situation)

            # Default shot type handling
            shot_type = details.get("shotType", "Unknown")

            # Determine the home team's defending side for this specific event
            x_coord = details.get("xCoord", None)
            y_coord = details.get("yCoord", None)
            season = int(str(game_data.get('id'))[:4])
            if season < 2019:
                zone_code = details.get("zoneCode", "")
                home_team_defending_side = get_home_team_defending_side(
                    x_coord, event_owner_team_id, home_team_id, zone_code, previous_defending_side
                )
                if home_team_defending_side is not None:
                    previous_defending_side = home_team_defending_side
            else:
                home_team_defending_side = event.get("homeTeamDefendingSide", None)
                if home_team_defending_side is not None:
                    previous_defending_side = home_team_defending_side

            # Calculate distance and angle to net
            distance_to_net, angle_to_net = None, None
            if x_coord is not None and y_coord is not None:
                distance_to_net, angle_to_net = calculate_distance_and_angle(
                    x_coord, y_coord, team_type, home_team_defending_side
                )
            else:
                print(
                    f"Game ID {game_data.get('id')}: Missing coordinates for event ID {event.get('eventId')}, skipping distance/angle calculation."
                )

            # Assign shooter_id based on event type
            shooter_id = (
                details.get("scoringPlayerId", "Unknown")
                if event_type == "goal"
                else details.get("shootingPlayerId", "Unknown")
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
                "home_team_defending_side": home_team_defending_side,
                "team_name": team_name,
                "team_type": team_type,
                "distance_to_net": distance_to_net,
                "angle_to_net": angle_to_net,
                "empty_net": empty_net_status,
                "strength_status": strength_status,
                "real_strength_home_vs_away": real_strength,
                "situation_code": situation_code,
                "shooter_id": shooter_id,
                "goalie_id": details.get("goalieInNetId", "Unknown"),
            }
            extracted_events.append(event_info)

        # Convert extracted events to a DataFrame
        df = pd.DataFrame(extracted_events)
        return df

    def json_to_csv_processing(self, seasons: List[int]):
        for season in seasons:
            src_season_dir_path = os.path.join(self.data_dir_path, "play_by_play", "json", f"{season}")
            dest_season_dir_path = os.path.join(self.data_dir_path, "play_by_play", "csv", f"{season}")
            os.makedirs(dest_season_dir_path, exist_ok=True)
            json_files_names = [json_file_name for json_file_name in os.listdir(src_season_dir_path) if json_file_name.endswith('.json')]
            for json_file_name in json_files_names:
                json_file_path = os.path.join(src_season_dir_path, json_file_name)
                csv_file_path = os.path.join(dest_season_dir_path, f"{json_file_name.split('.')[0]}.csv")
                with open(json_file_path, "r") as file:
                    dict_data = json.load(file)
                df_data = self.dict_to_data_frame_single_game(dict_data)
                df_data.to_csv(csv_file_path, index=False)
