from nhl_data_processor import NHLDataProcessor
from typing import List
import os
import json
import pandas as pd

class NHLDataExporter():
	def __init__(self):
        	self.data_dir_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        	os.makedirs(os.path.join(self.data_dir_path, "play_by_play", "csv"), exist_ok=True)

	def save_to_csv(self, df: pd.DataFrame) -> None:
		for game_id, group in df.groupby('game_id'):
			season = str(str(game_id)[:4])
			csv_file_path = os.path.join(self.data_dir_path, "play_by_play", "csv", f"{season}", f"{game_id}.csv")
			os.makedirs(os.path.join(self.data_dir_path, "play_by_play", "csv", f"{season}"), exist_ok=True)
			group.to_csv(csv_file_path, index=False)