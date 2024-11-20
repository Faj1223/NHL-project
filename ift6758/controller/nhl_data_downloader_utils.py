def generate_regular_season_game_id(season: int, game_number: int) -> str:
	"""
    Génère un identifiant unique pour un match de saison régulière.

    L'identifiant de match est une chaîne composée de trois parties :
    - L'année de la saison sous forme d'entier à 4 chiffres (par exemple, 2024).
    - Un identifiant de type de match fixe "02" pour les matchs de saison régulière.
    - Le numéro du match, formaté sous forme d'entier à 4 chiffres (par exemple, 0001).

    Entrées
    -------
    season : int
        L'année de début de la saison (par exemple, 2024 pour la saison 2024-2025).
    game_number : int
        Le numéro du match dans la saison régulière. Doit être un entier positif.

    Sortie
    ------
    str
        Une chaîne unique représentant l'identifiant du match, au format "AAAA02NNNN", où :
        - AAAA correspond à l'année de la saison.
        - 02 est le type de match pour les matchs de saison régulière.
        - NNNN est le numéro du match, complété par des zéros si nécessaire.

    Exemples
    --------
    >>> generate_regular_season_game_id(2024, 1)
    '2024020001'

    >>> generate_regular_season_game_id(2023, 123)
    '2023020123'
    """
	season_str = str(season)
	game_type_str = "02" # 02 is the game type for regular season games
	game_number_str = f"{game_number:04d}"
	return f"{season_str}{game_type_str}{game_number_str}"

def generate_playoff_game_id(season: int, round_num: int, matchup: int, game_num: int) -> str:
	"""
    Génère un identifiant unique pour un match de playoffs.

    L'identifiant du match est une chaîne composée de plusieurs parties :
    - L'année de la saison sous forme d'entier à 4 chiffres (par exemple, 2023).
    - Un identifiant fixe "03" pour indiquer qu'il s'agit d'un match de playoffs.
    - Une chaîne combinant :
        - Le numéro de la ronde des playoffs (01 pour la première ronde, jusqu'à 04 pour la finale).
        - Le numéro de l'affrontement (matchup) dans cette ronde.
        - Le numéro du match dans la série (entre 1 et 7).

    Entrées
    -------
    season : int
        L'année de début de la saison (par exemple, 2023 pour la saison 2023-2024).
    round_num : int
        Le numéro de la ronde des playoffs. Doit être compris entre 1 (première ronde) et 4 (finale).
    matchup : int
        Le numéro de l'affrontement dans la ronde spécifiée.
    game_num : int
        Le numéro du match dans la série. Doit être compris entre 1 et 7.

    Sortie
    ------
    str
        Une chaîne représentant l'identifiant unique du match de playoffs au format "AAAA03RRMGG", où :
        - AAAA est l'année de la saison.
        - 03 indique un match de playoffs.
        - RR est la ronde des playoffs.
        - M est l'affrontement (matchup) dans cette ronde.
        - GG est le numéro du match dans la série.

    Exemples
    --------
    >>> generate_playoff_game_id(2023, 1, 2, 3)
    '2023030123'
    
    >>> generate_playoff_game_id(2024, 4, 1, 7)
    '2024030417'
    """
	season_str = str(season)
	game_type_str = "03"
	game_number_str = f"{round_num:02d}{matchup}{game_num}"
	return f"{season_str}{game_type_str}{game_number_str}"