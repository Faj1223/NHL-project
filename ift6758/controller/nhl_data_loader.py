import os
import pandas as pd
from typing import List
import json

class NHLDataLoader():
	"""
	Classe dédiée à la gestion du chargement des données à partir de fichiers.
	"""

	def __init__(self):
		self.data_dir_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

	def load_json_file(self, game_id: str) -> dict:
		season = str(game_id[:4])
		json_file_path = os.path.join(self.data_dir_path, "play_by_play", "json", f"{season}", f"{game_id}.json")
		if not os.path.exists(json_file_path):
			print(f"Le fichier {game_id}.json n'existe pas.")
			return None
		else:
			with open(json_file_path, "r") as file:
				data = json.load(file)
			return data

	def load_csv_file(self, game_id: str) -> pd.DataFrame:
		season = str(game_id[:4])
		csv_file_path = os.path.join(self.data_dir_path, "play_by_play", "csv", f"{season}", f"{game_id}.csv")
		if not os.path.exists(csv_file_path):
			print(f"Le fichier {game_id}.csv n'existe pas.")
			return None
		else:
			return pd.read_csv(csv_file_path)

	def load_csv_file_only_shot_events(self, game_id: str) -> pd.DataFrame:
		season = str(game_id[:4])
		csv_file_path = os.path.join(self.data_dir_path, "play_by_play", "csv", f"{season}", f"{game_id}.csv")
		if not os.path.exists(csv_file_path):
			print(f"Le fichier {game_id}.csv n'existe pas.")
			return None
		else:
			return pd.read_csv(csv_file_path).query("event_type in ['shot-on-goal','goal']")

	def load_csv_files(self, seasons: List[int]) -> pd.DataFrame:
		"""
		Récupère les données play-by-play pour une ou plusieurs saisons.

		Parameters
		----------
		seasons : List[int]
			Une liste d'années représentant les saisons (par exemple, [2020, 2021]).

		Returns
		-------
		pd.DataFrame
			Un DataFrame contenant les données pour les saisons 'seasons'.
		"""
		all_df = []
		for season in seasons:
			season_dir_path = os.path.join(self.data_dir_path, "play_by_play", "csv", f"{season}")
			csv_files_names = [csv_file_name for csv_file_name in os.listdir(season_dir_path) if csv_file_name.endswith('.csv')]
			for csv_file_name in csv_files_names:
				csv_file_path = os.path.join(season_dir_path, csv_file_name)
				df = pd.read_csv(csv_file_path)
				all_df.append(df)
		combined_df = pd.concat(all_df, ignore_index=True)
		return combined_df

	def load_csv_files_only_shot_events(self, seasons: List[int]) -> pd.DataFrame:
		"""
		Récupère seulement les données play-by-play dont le type d'évènement est 'shot-on-goal' pour une ou plusieurs saisons.

		Parameters
		----------
		seasons : List[int]
			Une liste d'années représentant les saisons (par exemple, [2020, 2021]).

		Returns
		-------
		pd.DataFrame
			Un DataFrame contenant les données pour les saisons 'seasons'.
		"""
		all_df = []
		for season in seasons:
			season_dir_path = os.path.join(self.data_dir_path, "play_by_play", "csv", f"{season}")
			csv_files_names = [csv_file_name for csv_file_name in os.listdir(season_dir_path) if csv_file_name.endswith('.csv')]
			for csv_file_name in csv_files_names:
				csv_file_path = os.path.join(season_dir_path, csv_file_name)
				df = pd.read_csv(csv_file_path).query("event_type in ['shot-on-goal','goal']")
				all_df.append(df)
		combined_df = pd.concat(all_df, ignore_index=True)
		return combined_df

	def load_season_data(self, season_range):
		"""
		Load and concatenate only regular season data for a specified range of seasons.

		Parameters:
		- season_range: List or range of season years (e.g., range(2016, 2020)).2020 is not included

		Returns:
		- A Pandas DataFrame containing the combined regular season data for the given seasons.
		"""
		all_data = []

		for season in season_range:
			folder_name = f"{season}_CleanCSV"
			folder_path = os.path.join(self.DATA_DIR, folder_name)

			if os.path.exists(folder_path):
				csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".csv")]

				for file in csv_files:
					game_id = os.path.basename(file).split(".")[0]
					if game_id.startswith(str(season)) and game_id[4:6] == "02":  # Check game type "02"
						df = pd.read_csv(file)
						all_data.append(df)

				print(f"Loaded regular season data from folder: {folder_name}")
			else:
				print(f"Folder not found for season: {season} ({folder_name})")

		# Combine all data into a single DataFrame
		if all_data:
			combined_df = pd.concat(all_data, ignore_index=True)
			print(f"Combined regular season data for seasons {list(season_range)}.")
			return combined_df
		else:
			print("No regular season data found for the specified seasons.")
			return pd.DataFrame()