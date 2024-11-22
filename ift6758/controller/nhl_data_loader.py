import os
import pandas as pd
from typing import List
import json
from sklearn.feature_selection import SelectKBest, f_classif

def is_a_regular_season_game(game_id: str) -> bool:
	game_type = game_id[4:6]
	return game_type == "02" # 02 is the game type for regular season games

def is_a_playoff_game(game_id: str) -> bool:
	game_type = game_id[4:6]
	return game_type == "03" # 03 is the game type for playoff games

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

	def load_csv_files_only_regular_season_games(self, seasons: List[int]) -> pd.DataFrame:
		all_df = []
		for season in seasons:
			season_dir_path = os.path.join(self.data_dir_path, "play_by_play", "csv", f"{season}")
			csv_files_names = [csv_file_name for csv_file_name in os.listdir(season_dir_path) if csv_file_name.endswith('.csv')and is_a_regular_season_game(csv_file_name.split('.')[0])]
			for csv_file_name in csv_files_names:
				csv_file_path = os.path.join(season_dir_path, csv_file_name)
				df = pd.read_csv(csv_file_path)
				all_df.append(df)
		combined_df = pd.concat(all_df, ignore_index=True)
		return combined_df
	
	def load_csv_files_only_playoff_games(self, seasons: List[int]) -> pd.DataFrame:
		all_df = []
		for season in seasons:
			season_dir_path = os.path.join(self.data_dir_path, "play_by_play", "csv", f"{season}")
			csv_files_names = [csv_file_name for csv_file_name in os.listdir(season_dir_path) if csv_file_name.endswith('.csv')and is_a_playoff_game(csv_file_name.split('.')[0])]
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

	def load_csv_files_only_regular_season_games_only_shot_events(self, seasons: List[int]) -> pd.DataFrame:
		all_df = []
		for season in seasons:
			season_dir_path = os.path.join(self.data_dir_path, "play_by_play", "csv", f"{season}")
			csv_files_names = [csv_file_name for csv_file_name in os.listdir(season_dir_path) if csv_file_name.endswith('.csv') and is_a_regular_season_game(csv_file_name.split('.')[0])]
			for csv_file_name in csv_files_names:
				csv_file_path = os.path.join(season_dir_path, csv_file_name)
				df = pd.read_csv(csv_file_path).query("event_type in ['shot-on-goal','goal']")
				all_df.append(df)
		combined_df = pd.concat(all_df, ignore_index=True)
		return combined_df
	
	def load_csv_files_only_playoff_games_only_shot_events(self, seasons: List[int]) -> pd.DataFrame:
		all_df = []
		for season in seasons:
			season_dir_path = os.path.join(self.data_dir_path, "play_by_play", "csv", f"{season}")
			csv_files_names = [csv_file_name for csv_file_name in os.listdir(season_dir_path) if csv_file_name.endswith('.csv') and is_a_playoff_game(csv_file_name.split('.')[0])]
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
			folder_name = f"{season}"
			folder_path = os.path.join(self.data_dir_path, "play_by_play", "csv", f"{season}")

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

	def load_old_csv_files(self, seasons: List[int]) -> pd.DataFrame:
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
			season_dir_path = os.path.join(self.data_dir_path, "play_by_play","json", f"{season}_CleanCSV")
			csv_files_names = [csv_file_name for csv_file_name in os.listdir(season_dir_path) if csv_file_name.endswith('.csv')]
			for csv_file_name in csv_files_names:
				csv_file_path = os.path.join(season_dir_path, csv_file_name)
				df = pd.read_csv(csv_file_path)
				all_df.append(df)
		combined_df = pd.concat(all_df, ignore_index=True)
		return combined_df

	@staticmethod
	def get_optimal_features_dataframe(df, target_column='is_goal', k=5):
		"""
        Sélectionne les caractéristiques optimales à l'aide de SelectKBest.

        Parameters:
        - df: DataFrame brut avec les colonnes pertinentes.
        - target_column: Nom de la colonne cible (par défaut: 'is_goal').
        - k: Nombre de caractéristiques à sélectionner (par défaut: 5).

        Returns:
        - DataFrame réduit avec les caractéristiques optimales et la cible.
        """
		# Supprime les lignes avec des valeurs manquantes
		df = df.dropna()

		# Séparer les caractéristiques et la cible
		X = df.drop(columns=[target_column])
		y = df[target_column]

		# Sélection des k meilleures caractéristiques
		selector = SelectKBest(score_func=f_classif, k=k)
		selector.fit(X, y)  # Pas besoin de transformer ici, car nous utilisons les colonnes

		# Colonnes sélectionnées
		selected_features = X.columns[selector.get_support()]
		print(f"Caractéristiques sélectionnées : {list(selected_features)}")

		# Construire le DataFrame final
		df_selected = X[selected_features].copy()
		df_selected[target_column] = y
		return df_selected
