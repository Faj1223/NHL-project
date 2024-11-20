import os
import sys

if os.path.join(os.path.dirname(os.path.dirname(__file__)), "controller") not in sys.path:
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "controller"))

from nhl_data_loader import NHLDataLoader
from nhl_data_cleaner import NHLDataCleaner
from nhl_data_exporter import NHLDataExporter

# À compléter
seasons = [2016, 2017, 2018, 2019, 2020, 2021]

data_dir_path = os.path.join(os.path.dirname(os.getcwd()), "data")

loader = NHLDataLoader()
cleaner = NHLDataCleaner()
exporter = NHLDataExporter()

df = loader.load_csv_files(seasons)

df = cleaner.clean(df)

exporter.save_to_csv(df)
