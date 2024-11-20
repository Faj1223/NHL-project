import os
import sys

if os.path.join(os.path.dirname(os.path.dirname(__file__)), "controller") not in sys.path:
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "controller"))

from nhl_data_processor import NHLDataProcessor
from nhl_data_loader import NHLDataLoader
from nhl_data_exporter import NHLDataExporter

# À compléter
seasons = [2016, 2017, 2018, 2019, 2020, 2021]

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
        df = processor.dictionary_to_dataframe_single_game(dictionary)
        exporter.save_to_csv(df)
