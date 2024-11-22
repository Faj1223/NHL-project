import os
import sys

if os.path.join(os.path.dirname(os.path.dirname(__file__)), "controller") not in sys.path:
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "controller"))

from nhl_data_loader import NHLDataLoader
from nhl_data_processor import NHLDataProcessor
from nhl_data_exporter import NHLDataExporter

# À compléter
seasons = [2016, 2017, 2018, 2019, 2020, 2021]

data_dir_path = os.path.join(os.path.dirname(os.getcwd()), "data")

loader = NHLDataLoader()
processor = NHLDataProcessor()
exporter = NHLDataExporter()

df = loader.load_csv_files(seasons)

df = processor.sort(df)

exporter.save_to_csv(df)
