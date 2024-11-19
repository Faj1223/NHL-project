import numpy as np

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

#Helper function to calculate distance and angle
def calculate_distance_and_angle(x_coord, y_coord, team_type, home_team_defending_side):
	"""
	Calculate the distance and angle to the net based on the event's coordinates,
	team type, and the home team's defending side.

	Parameters:
	- x_coord: x-coordinate of the event.
	- y_coord: y-coordinate of the event.
	- team_type: Indicates whether the event involves the home or away team ('home' or 'away').
	- home_team_defending_side: The side ("left" or "right") the home team is defending.

	Returns:
	- distance_to_net: Distance from the event location to the net.
	- angle_to_net: Angle from the event location to the net (in degrees).
	"""
	if x_coord is None or y_coord is None or home_team_defending_side not in ["left", "right"]:
		return np.nan, np.nan  # Return NaN for invalid inputs

	# Determine net_x based on team type and defending side
	if team_type == "home":
		net_x = 89 if home_team_defending_side == "left" else -89
	elif team_type == "away":
		net_x = -89 if home_team_defending_side == "left" else 89
	else:
		return np.nan, np.nan  # Invalid team type

	net_y = 0  # Assume the net is centered on the y-axis

	# Calculate distance and angle
	distance_to_net = ((x_coord - net_x) ** 2 + (y_coord - net_y) ** 2) ** 0.5
	angle_to_net = np.arctan2(y_coord - net_y, net_x - x_coord) * (180 / np.pi)  # Convert to degrees

	return distance_to_net, angle_to_net