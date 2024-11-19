import os
import numpy as np
import pandas as pd
import requests
import json
from IPython.display import display
import ipywidgets as widgets
from nhl_data_downloader_utils import generate_regular_season_game_id, generate_playoff_game_id

class NHLDataDownloader():
    """
    Classe pour télécharger les données des matchs de la Ligue Nationale de Hockey (NHL) à partir d'une API Web.

    Cette classe permet de télécharger les données détaillées des matchs de la saison régulière ainsi que des playoffs 
    de la NHL, et de les sauvegarder localement au format JSON. Elle offre des fonctionnalités pour télécharger les données 
    pour une saison complète ou pour une plage de saisons, tout en gérant les erreurs de téléchargement et en affichant une 
    barre de progression dans une interface Jupyter.

    Attributs
    ---------
    base_url : str
        L'URL de base de l'API pour récupérer les données des matchs (https://api-web.nhle.com/v1/gamecenter/).

    data_dir_path : str
        Le chemin vers le répertoire où les données téléchargées seront stockées localement.

    suffix_url : str
        Le suffixe de l'URL utilisé pour accéder aux détails des matchs ("/play-by-play").

    progress_bar : ipywidgets.FloatProgress
        Un widget de barre de progression pour afficher l'état de téléchargement dans une interface Jupyter.
    
    Méthodes
    --------
    __init__ :
        Initialise les attributs de la classe et crée les répertoires nécessaires pour le stockage des données téléchargées.

    download_game_data :
        Télécharge les données d'un match spécifique et les sauvegarde localement.
    
    download_regular_season :
        Télécharge les données de tous les matchs de la saison régulière pour une saison donnée.
    
    download_playoff_series :
        Télécharge les données des matchs d'une série de playoffs.
    
    download_playoffs :
        Télécharge les données de l'ensemble des playoffs pour une saison donnée, en gérant les différents tours et confrontations.
    
    download_all_seasons_play_by_play :
        Télécharge les données des matchs réguliers et des playoffs pour une plage de saisons spécifiée.
    """

    def __init__(self):
        """
        Initialise les paramètres et crée les répertoires nécessaires pour télécharger et stocker les données des matchs.

        Cette méthode configure l'URL de base de l'API, le chemin du répertoire de stockage des données locales, 
        et crée les sous-dossiers nécessaires pour organiser les fichiers JSON. Elle initialise également la barre 
        de progression qui sera utilisée pour afficher l'état du téléchargement dans une interface Jupyter.

        Effets secondaires
        ------------------
        - Crée les répertoires nécessaires dans `self.data_dir_path` s'ils n'existent pas.
        - Initialise la barre de progression pour le suivi de l'état du téléchargement des données.

        Exemples
        --------
        >>> downloader = NHLDataDownloader()
        >>> # La classe est maintenant prête à télécharger les données des matchs
        """
        self.base_url = "https://api-web.nhle.com/v1/gamecenter/"
        self.data_dir_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        self.suffix_url = "/play-by-play"
        self.progress_bar = widgets.FloatProgress(value=0.0, min=0.0, max=1.0, description='Loading:',)
        os.makedirs(self.data_dir_path, exist_ok=True)
        os.makedirs(os.path.join(self.data_dir_path, "play_by_play"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir_path, "play_by_play", "json"), exist_ok=True)

    def download_game_data(self, game_id: str, season: int) -> None:
        """
        Télécharge les données pour un match spécifique et les stocke localement.

        Cette fonction télécharge les données au format JSON pour un identifiant de match donné 
        (game ID) et les sauvegarde dans un répertoire local organisé par saison. Si le fichier 
        correspondant existe déjà dans le cache local, la fonction ne télécharge pas à nouveau les données.

        Entrées
        -------
        game_id : str
            L'identifiant unique du match (par exemple, "2023020456").

        season : int
            L'année de début de la saison (par exemple, 2023 pour la saison 2023-2024).

        Sortie
        ------
        None
            Cette fonction ne retourne rien. Les données sont sauvegardées localement dans un fichier JSON.

        Effets secondaires
        ------------------
        - Crée les dossiers nécessaires dans le répertoire `self.data_dir_path` si ceux-ci n'existent pas.
        - Télécharge les données depuis l'URL construite avec `game_id` et `self.base_url`.
        - Sauvegarde les données dans un fichier JSON au chemin suivant : 'self.data_dir_path/play_by_play/json/{season}/{game_id}.json'.

        Notes
        -----
        - Si le fichier existe déjà dans le cache local, un message est affiché et aucune action supplémentaire n'est effectuée.
        - Si le téléchargement échoue ou si les données ne peuvent pas être décodées, un message d'erreur est affiché.

        Exemples
        --------
        >>> downloader = NHLDataDownloader()
        >>> downloader.download_game_data("2023020456", 2023)
        Game data for 2023020456 already exists in local cache.

        >>> downloader.download_game_data("2023020456", 2024)
        Failed to download data for game 2023020456.
        """
        game_file_path = os.path.join( self.data_dir_path, "play_by_play", "json", f"{season}", f"{game_id}.json")
        os.makedirs(os.path.join(self.data_dir_path, "play_by_play", "json", f"{season}"), exist_ok=True)
        if os.path.exists(game_file_path):
            print(f"Game data for {game_id} already exists in local cache.")
            return
        url = f"{self.base_url}{game_id}{self.suffix_url}"
        response = requests.get(url)
        if response.status_code == 200:
            try:
                data = json.loads(response.text)
                with open(game_file_path, "w") as file:
                    json.dump(data, file)
            except json.JSONDecodeError as e:
                print(f"Failed to decode cleaned JSON: {e}")
                return
        else:
            print(f"Failed to download data for game {game_id}.")
            return

    def download_regular_season(self, season: int, total_games: int = 1353, output_widget=None) -> None:
        """
        Télécharge les données de tous les matchs de la saison régulière pour une saison donnée.

        Cette fonction télécharge les données de chaque match de saison régulière pour une saison donnée, 
        en créant les identifiants des matchs de manière séquentielle et en utilisant la méthode 
        `download_game_data` pour obtenir et sauvegarder les données localement. Une barre de progression 
        est affichée pour indiquer l'avancement du processus.

        Entrées
        -------
        season : int
            L'année de début de la saison (par exemple, 2023 pour la saison 2023-2024).

        total_games : int, optionnel
            Le nombre total de matchs de la saison régulière. Par défaut, 1353.

        output_widget : widget ou None, optionnel
            Widget d'affichage optionnel pour afficher la barre de progression dans une interface Jupyter.
            Si `None`, la barre de progression est affichée par défaut.

        Sortie
        ------
        None
            Cette fonction ne retourne rien. Les données des matchs sont sauvegardées localement.

        Effets secondaires
        ------------------
        - Met à jour la barre de progression (`self.progress_bar`) à chaque téléchargement de match.
        - Télécharge et stocke les données des matchs dans les dossiers définis par `self.download_game_data`.

        Notes
        -----
        - Les fichiers JSON des matchs sont sauvegardés localement selon la structure définie dans 
        self.download_game_data`.
        - Si un widget de sortie est fourni, la barre de progression est affichée dans ce widget.
        - Cette fonction suppose que `self.download_game_data` gère les éventuelles erreurs de téléchargement.

        Exemples
        --------
        >>> downloader = NHLDataDownloader()
        >>> downloader.download_regular_season(2023)
        Téléchargement de 1353 matchs pour la saison 2023...

        >>> downloader.download_regular_season(2024, total_games=1000, output_widget=my_output_widget)
        Barre de progression affichée dans le widget spécifié.
        """
        self.progress_bar.value = 0
        if output_widget == None:
            display(self.progress_bar)
        else:
            with output_widget:
                display(self.progress_bar)
        for game_num in range(1, total_games+1):
            game_id = generate_regular_season_game_id(season, game_num)
            self.download_game_data(game_id, season)
            self.progress_bar.value = (game_num-1) / total_games

    def download_playoff_series(self, season: int, round_num: int, matchup: int, total_games: int = 7) -> None:
        """
        Télécharge les données de tous les matchs d'une série de playoffs.

        Cette fonction télécharge les données de chaque match d'une série de playoffs pour une saison donnée,
        en générant l'identifiant de chaque match à partir de l'année de la saison, du numéro du tour, 
        du numéro de la confrontation et du numéro du match dans la série. La fonction utilise la méthode
        `download_game_data` pour télécharger et sauvegarder les données de chaque match.

        Entrées
        -------
        season : int
            L'année de début de la saison (par exemple, 2023 pour la saison 2023-2024).

        round_num : int
            Le numéro du tour des playoffs (par exemple, 1 pour le premier tour, 2 pour le second, etc.).

        matchup : int
            Le numéro de la confrontation dans le tour des playoffs (par exemple, 1 pour la première confrontation, etc.).

        total_games : int, optionnel
            Le nombre total de matchs dans la série de playoffs. Par défaut, 7.

        Sortie
        ------
        None
            Cette fonction ne retourne rien. Les données des matchs sont téléchargées et sauvegardées localement.

        Effets secondaires
        ------------------
        - Télécharge et stocke les données des matchs dans un répertoire local, un par un, en utilisant `self.download_game_data`.

        Notes
        -----
        - Les matchs sont téléchargés en fonction du numéro de tour et de la confrontation dans la série de playoffs.
        - La fonction suppose que la méthode `self.download_game_data` gère les erreurs de téléchargement et de stockage.
        - Par défaut, la série comprend jusqu'à 7 matchs, mais le nombre de matchs peut être ajusté via le paramètre `total_games`.

        Exemples
        --------
        >>> downloader = NHLDataDownloader()
        >>> downloader.download_playoff_series(2023, 1, 1)
        Téléchargement des matchs de la série de playoffs pour la saison 2023, premier tour, confrontation 1.

        >>> downloader.download_playoff_series(2024, 2, 3, total_games=5)
        Téléchargement des matchs de la série de playoffs pour la saison 2024, second tour, confrontation 3, avec 5 matchs.
        """
        for game_num in range(1, total_games+1):
            game_id = generate_playoff_game_id(season, round_num, matchup, game_num)
            self.download_game_data(game_id, season)

    def download_playoffs(self, season: int, output_widget=None) -> None:
        """
        Télécharge les données de l'ensemble des playoffs pour une saison donnée, en ajustant dynamiquement les confrontations pour chaque tour :
        - Tour 1 : 8 confrontations
        - Tour 2 : 4 confrontations
        - Tour 3 : 2 confrontations
        - Tour 4 : 1 confrontation (Finale de la Coupe Stanley)

        Entrées
        -------
        season : int
            L'année de début de la saison (par exemple, 2023 pour la saison 2023-2024).

        output_widget : widget ou None, optionnel
            Widget d'affichage optionnel pour afficher la barre de progression dans une interface Jupyter.
            Si `None`, la barre de progression est affichée par défaut.

        Sortie
        ------
        None
            Cette fonction ne retourne rien. Les données des matchs des playoffs sont téléchargées et sauvegardées localement.

        Effets secondaires
        ------------------
        - Télécharge et stocke les données des matchs de playoffs dans un répertoire local.
        - Met à jour la barre de progression (`self.progress_bar`) à chaque téléchargement de série de playoffs.

        Notes
        -----
        - Cette fonction gère les 4 tours des playoffs avec un nombre de confrontations décroissant par tour : 8 au premier tour, 4 au deuxième,
        2 au troisième et 1 pour la finale de la Coupe Stanley.
        - La fonction suppose que la méthode `self.download_playoff_series` gère correctement le téléchargement des matchs pour chaque série.
        - Si un widget de sortie est fourni, la barre de progression est affichée dans ce widget.

        Exemples
        --------
        >>> downloader = NHLDataDownloader()
        >>> downloader.download_playoffs(2023)
        Téléchargement des matchs de playoffs pour la saison 2023 avec gestion des 4 tours.

        >>> downloader.download_playoffs(2024, output_widget=my_output_widget)
        Téléchargement des matchs de playoffs pour la saison 2024, avec la barre de progression affichée dans le widget spécifié.
        """
        self.progress_bar.value = 0.0
        if output_widget == None:
            display(self.progress_bar)
        else:
            with output_widget:
                display(self.progress_bar)
        rounds_matchups = {1: 8, 2: 4, 3: 2, 4: 1}
        for round_num, matchups in rounds_matchups.items():
            for matchup in range(1, matchups + 1):
                self.download_playoff_series(season, round_num, matchup)
                self.progress_bar.value += (1/matchups) / matchups

    def download_all_seasons_play_by_play(self, start_season: int, end_season: int) -> None:
        """
        Télécharge toutes les données des matchs réguliers et des playoffs pour une plage de saisons.

        Cette fonction télécharge les données pour chaque saison dans la plage spécifiée, 
        en appelant successivement `self.download_regular_season` pour les matchs de saison régulière
        et `self.download_playoffs` pour les matchs de playoffs pour chaque saison.

        Entrées
        -------
        start_season : int
            L'année de début de la plage de saisons (par exemple, 2023 pour commencer la plage à 2023).
            
        end_season : int
            L'année de fin de la plage de saisons (par exemple, 2024 pour finir la plage à 2024).

        Sortie
        ------
        None
            Cette fonction ne retourne rien. Les données des matchs de saison régulière et de playoffs 
            sont téléchargées et stockées localement pour chaque saison dans la plage.

        Effets secondaires
        ------------------
        - Télécharge et sauvegarde les données des matchs pour chaque saison spécifiée.
        - La méthode `self.download_regular_season` est appelée pour chaque saison, suivie de `self.download_playoffs`.

        Notes
        -----
        - La fonction traite toutes les saisons entre `start_season` et `end_season` inclusivement.
        - Si une saison n'a pas de données disponibles ou si une erreur survient lors du téléchargement, 
        celle-ci est ignorée et le processus continue avec les autres saisons.

        Exemples
        --------
        >>> downloader = NHLDataDownloader()
        >>> downloader.download_all_seasons_play_by_play(2020, 2022)
        Téléchargement des matchs de saison régulière et de playoffs pour les saisons 2020, 2021 et 2022.

        >>> downloader.download_all_seasons_play_by_play(2023, 2024)
        Téléchargement des matchs de saison régulière et de playoffs pour les saisons 2023 et 2024.
        """
        for season in range(start_season, end_season+1):
            self.download_regular_season(season)
            self.download_playoffs(season)