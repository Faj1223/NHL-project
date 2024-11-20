import os
import sys

if os.path.join(os.path.dirname(os.path.dirname(__file__)), "controller") not in sys.path:
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "controller"))

from nhl_data_processor import NHLDataProcessor
from nhl_data_loader import NHLDataLoader
from nhl_data_exporter import NHLDataExporter

# À compléter (par exemple seasons = [2016, 2017, 2018, 2019, 2020, 2021])
seasons = [2017]

data_dir_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

loader = NHLDataLoader()
processor = NHLDataProcessor()
exporter = NHLDataExporter()

for season in seasons:
    csv_files_season_dir_path = os.path.join(data_dir_path, "play_by_play", "json", f"{season}")
    json_files_names = [json_file_name for json_file_name in os.listdir(csv_files_season_dir_path) if json_file_name.endswith('.json')]
    for json_file_name in json_files_names:
        game_id = json_file_name.split('.')[0]
        dictionary = loader.load_json_file(game_id)
        df = processor.dictionary_to_data_frame_single_game(dictionary)
        exporter.save_to_csv(df)

df = loader.load_csv_files(seasons)

df = processor.add_last_event_type_column(df)
df = processor.add_last_x_coord_column(df)
df = processor.add_last_y_coord_column(df)
df = processor.add_time_since_last_event_column(df)
df = processor.add_distance_from_last_event_column(df)
df = processor.add_rebound_column(df)
df = processor.add_home_team_defending_side_column(df)
df = processor.add_shooting_distance_column(df)
df = processor.add_shot_angle_column(df)
df = processor.add_rebound_angle_column(df)
df = processor.add_speed_from_last_event_column(df)

exporter.save_to_csv(df)