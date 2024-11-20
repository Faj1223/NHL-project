import numpy as np
import math
import pandas as pd

#Helper function to parse the situation code
def parse_situation_code(situation_code):
        """
        Parse the situation code and return a dictionary with the parsed values.
        1st: away goalie (1=in net, 0=pulled)
        2nd: number of away skaters
        3rd: number of home skaters
        4th: home goalie (1=in net, 0=pulled)
        """
        if len(situation_code) == 4:
            away_goalie_in_net = situation_code[0] == '1'
            away_skaters = int(situation_code[1])
            home_skaters = int(situation_code[2])
            home_goalie_in_net = situation_code[3] == '1'
            return {
                'away_goalie_in_net': away_goalie_in_net,
                'away_skaters': away_skaters,
                'home_skaters': home_skaters,
                'home_goalie_in_net': home_goalie_in_net
            }

#Helper function to get the strength status
def get_strength_status(parsed_situation):
	"""
	Determine whether the goal or shot occurred during equal strength, power play, or penalty kill.

	Arguments:
	- parsed_situation: Dictionary containing details of skaters and goalie presence.

	Returns:
	- A string indicating 'Equal Strength', 'Power Play', or 'Penalty Kill'
	"""
	away_skaters = parsed_situation.get('away_skaters', 0)
	home_skaters = parsed_situation.get('home_skaters', 0)

	if home_skaters == away_skaters:
		return "Equal Strength"
	elif home_skaters > away_skaters:
		return "Power Play"  # Home team has more skaters
	else:
		return "Penalty Kill"  # Home team has fewer skaters

#Helper function to determine if the net is empty
def is_net_empty_goal(team_type, parsed_situation):
	"""
	Determine if the net is empty based on the team type (home or away).

	Args:
	- team_type: A string indicating whether the team is 'home' or 'away'
	- parsed_situation: A dictionary containing parsed situation details (goalie in net, skaters, etc.)

	Returns:
	- True if the net is empty for the given team, False otherwise.
	"""
	# Ensure NaN values are treated as False (0)
	away_goalie_in_net = parsed_situation.get('away_goalie_in_net', False) or False
	home_goalie_in_net = parsed_situation.get('home_goalie_in_net', False) or False

	if team_type == "home":
		return not away_goalie_in_net  # If away goalie is not in net, it's an empty net for home
	elif team_type == "away":
		return not home_goalie_in_net  # If home goalie is not in net, it's an empty net for away
	return False  # Default: net is not empty if team type is unknown

#Helper function to determine the home team's defending side
def get_home_team_defending_side(x_coord, event_owner_team_id, home_team_id, zone_code, previous_defending_side):
	"""
	Determines which side ("left" or "right") is the home team’s defensive zone
	based on the x-coordinate, event owner, and zone code.

	Parameters:
	- x_coord: x-coordinate of the event.
	- event_owner_team_id: ID of the team associated with the event.
	- home_team_id: The ID of the home team.
	- zone_code: Zone code indicating if the event is offensive ("O") or defensive ("D").

	Returns:
	- "left" or "right" indicating the defensive side of the home team for the event.
	"""
	if x_coord is None:
		print(
			f"x_coord is None for the event with event_owner_team_id: {event_owner_team_id} and zone_code: {zone_code}. Using previous defending side.")
		return previous_defending_side
	# Case 1: Event involves the home team
	if event_owner_team_id == home_team_id:
		if zone_code == "O":  # Offensive zone for home team
			# Current side is offensive, so the opposite side is defensive
			if x_coord > 0:
				return "left"  # Right is offensive, so left is defensive
			elif x_coord < 0:
				return "right"  # Left is offensive, so right is defensive
		elif zone_code == "D":  # Defensive zone for home team
			# Current side is defensive
			if x_coord > 0:
				return "right"  # Right is defensive
			elif x_coord < 0:
				return "left"  # Left is defensive

	# Case 2: Event involves the away team
	else:
		if zone_code == "D":  # Defensive zone for home team
			# Current side is defensive
			if x_coord > 0:
				return "left"  # Right is defensive for away, so left is defensive for home
			elif x_coord < 0:
				return "right"  # Left is defensive for away, so right is defensive for home
		elif zone_code == "O":  # Offensive zone for away team
			# Current side is offensive
			if x_coord > 0:
				return "right"  # Right is offensive for away, so right is defensive for home
			elif x_coord < 0:
				return "left"  # Left is offensive for away, so left is defensive for home
	if zone_code == "N":
		return previous_defending_side
		# Default return if no other conditions match
	print(
		f"No matching conditions for event_owner_team_id: {event_owner_team_id}, home_team_id: {home_team_id}, zone_code: {zone_code}. Returning None.")
	return None

def calculate_duration_mm_ss(time1: str, time2: str) -> int:
    # Convertir les durées en secondes
    minutes1, seconds1 = map(int, time1.split(":"))
    minutes2, seconds2 = map(int, time2.split(":"))
    
    total_seconds1 = minutes1 * 60 + seconds1
    total_seconds2 = minutes2 * 60 + seconds2
    
    # Calculer la différence en secondes
    duration_seconds = total_seconds2 - total_seconds1
    
    return duration_seconds

def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Calcule la distance euclidienne entre deux points (x1, y1) et (x2, y2).

    :param x1: Coordonnée x du premier point
    :param y1: Coordonnée y du premier point
    :param x2: Coordonnée x du deuxième point
    :param y2: Coordonnée y du deuxième point
    :return: Distance euclidienne entre les deux points
    """
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_shooting_distance(tir: pd.Series) -> float:
	shooter_coords = (tir['x_coord'], tir['y_coord'])

	if tir['home_team_defending_side'] == 'left':
		home_goalie_coords = (-100, 0)
		away_goalie_coords = (100, 0)
	else:
		home_goalie_coords = (100, 0)
		away_goalie_coords = (-100, 0)

	if tir['team_type'] == 'home':
		return euclidean_distance(shooter_coords[0], shooter_coords[1], away_goalie_coords[0], away_goalie_coords[1])
	else:
		return euclidean_distance(shooter_coords[0], shooter_coords[1], home_goalie_coords[0], home_goalie_coords[1])

def compute_angle(x_shot: float, y_shot: float, x_net: float, y_net: float) -> float:
    delta_x = x_shot - x_net
    delta_y = y_shot - y_net
    if (delta_y == 0):
        return 90
    angle = np.arctan(delta_x / delta_y)
    if angle < 0:
	    angle = np.pi + angle
    return angle * (180 / np.pi)

def compute_angle_row(row) -> float:
    if row['home_team_defending_side'] == 'left':
        home_goalie_coords = (-100, 0)
        away_goalie_coords = (100, 0)
    else:
        home_goalie_coords = (100, 0)
        away_goalie_coords = (-100, 0)
    if row['team_type'] == 'home':
        return compute_angle(row['x_coord'], row['y_coord'], away_goalie_coords[0], away_goalie_coords[1])
    else:
        return compute_angle(row['x_coord'], row['y_coord'], home_goalie_coords[0], home_goalie_coords[1])