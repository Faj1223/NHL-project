import ipywidgets as widgets
from IPython.display import display, Markdown
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
import os

import ift6758
import sys
import importlib
from multiprocessing.util import debug

sys.path.append('..')
from ift6758.controller.nhl_data_downloader import NHLDataDownloader
#import controller.nhl_data_downloader

importlib.reload(ift6758.controller.nhl_data_downloader)
#from controller.nhl_data_downloader import NHLDataDownloader

class RinkEventDebugTool:
    def __init__(self, event_types):
        # Dowloader
        self.downloader = NHLDataDownloader()

        # Event types to care about when listing events in tool
        self.event_types = event_types 
        
        # Widgets
        self.season_dropdown = widgets.Dropdown(value='Select a year', options=['Select a year'] + [str(year) for year in range(2016, 2024)], description='Season')
        self.season_type_dropdown = widgets.Dropdown(options=['Regular Season', 'Playoffs'], description='Season Type')
        self.game_slider = widgets.IntSlider(min=0, description='Game ID')
        self.event_slider = widgets.IntSlider(min=0, description='Event ID')

        # Widgets Outputs (to display the rink image and event info)
        self.loading_ouput = widgets.Output()
        self.rink_image_output = widgets.Output()
        self.event_info_output = widgets.Output()
        self.game_info_output = widgets.Output()

        # Retrieve rink image
        current_dir = os.getcwd()
        rink_img_path = os.path.join(current_dir,'..', 'data', 'Images', 'nhl_rink.png')
        """ Normalize the path to avoid issues with '..'correct path for the rink image """
        rink_img_path = os.path.normpath(rink_img_path) 
        self.rink_img = mpimg.imread(rink_img_path)

        # Layout
        self.layout = widgets.VBox([
            self.season_dropdown, 
            self.season_type_dropdown,
            self.game_slider,
            self.game_info_output,  # Insert game info between the game and event sliders
            self.event_slider,
            self.loading_ouput,
            self.rink_image_output,  # Rink image after the event slider
            self.event_info_output  # Event info after the rink image
        ])
        
        # Games
        self.all_games_from_selection = {} # Contains all games from regular or playoff given a selected season
        self.filtered_events = {} # Contains events from selected season given the list of event types we care about


    # Function to plot event coordinates on the rink image
    # x,y         -> coord of the event
    # event       -> object containing all the event info
    # home_abbrev -> abbreviation of home's team name
    # away_abbrev -> abbreviation of away's team name
    def plot_event_on_rink(self, x, y, event, home_abbrev, away_abbrev):
        # Get event details for the title
        event_desc = event.get("typeDescKey", "Unknown Event")
        event_time = event.get("timeInPeriod", "Unknown Time")
        period_number = event.get("periodDescriptor", {}).get("number", "Unknown Period")
    
        # Construct the title string (similar to the example picture)
        title = f"{event_desc}\n{event_time} P-{period_number}"
    
        with self.rink_image_output:
            self.rink_image_output.clear_output()  # Clear the previous plot
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(self.rink_img, extent=[-100, 100, -42.5, 42.5])
            if x is not None and y is not None:
                ax.scatter(x, y, color='blue', label="Event location")
                ax.legend()
            ax.set_title(title)
            if period_number == 1 or period_number == 3:
                ax.text(-90, 45, away_abbrev, fontsize=12, ha='center', va='center')  # Away team on the left
                ax.text(90, 45, home_abbrev, fontsize=12, ha='center', va='center')  # Home team on the right
            elif period_number == 2:
                ax.text(90, 45, away_abbrev, fontsize=12, ha='center', va='center') # Away team on the right
                ax.text(-90, 45, home_abbrev, fontsize=12, ha='center', va='center') #  Home team on the left
            plt.show()


    # Function to format and return game information
    def get_game_info(self, game_data):   
        game_start_time = game_data.get("startTimeUTC", "Unknown Date")
        home_abbrev = game_data.get("homeTeam", {}).get("abbrev", "Unknown Home Abbrev")
        away_abbrev = game_data.get("awayTeam", {}).get("abbrev", "Unknown Away Abbrev")
        home_name = game_data.get("homeTeam", {}).get("name").get("default", "Unknown Home Name")
        away_name = game_data.get("awayTeam", {}).get("name").get("default", "Unknown Home Name")
        home_id = game_data.get("homeTeam", {}).get("id", 0)
        away_id = game_data.get("awayTeam", {}).get("id", 0)
        home_goals = game_data.get("homeTeam", {}).get("score", 0)
        away_goals = game_data.get("awayTeam", {}).get("score", 0)
        home_sog = game_data.get("homeTeam", {}).get("sog", 0)
        away_sog = game_data.get("awayTeam", {}).get("sog", 0)
    
        # Format the game state (period, overtime, etc.)
        game_period = game_data.get("periodDescriptor", {}).get("periodType", "Unknown")
        
        # Pretty display of game info
        game_info = f"""
        {game_start_time}
        Game ID: {game_data.get('id', 'Unknown')}; {home_abbrev} (home) vs {away_abbrev} (away)
    
        {game_period}
    
        {'':<15}{'Home'.ljust(20)}{'Away'.ljust(20)}
        {'Teams:'.ljust(15)}{home_name.ljust(20)}{away_name.ljust(20)}
        {'Team ID:'.ljust(15)}{str(home_id).ljust(20)}{str(away_id).ljust(20)}
        {'Goals:'.ljust(15)}{str(home_goals).ljust(20)}{str(away_goals).ljust(20)}
        {'SoG:'.ljust(15)}{str(home_sog).ljust(20)}{str(away_sog).ljust(20)}
        """
        # Output game info in Markdown for better formatting
        with self.game_info_output:
            self.game_info_output.clear_output(wait=True)
            display(Markdown(f"{game_info}"))
            
        return home_abbrev, away_abbrev


    # Function to display event details in JSON format
    def display_event_info(self, event):
        event_info_json = json.dumps(event, indent=4)
        with self.event_info_output:
            self.event_info_output.clear_output(wait=True)
            display(Markdown(f"```json\n{event_info_json}\n```"))
            
            
    # Function to download data based on season and season type
    def on_season_or_type_change(self, change):
        if self.season_dropdown.value == 'Select a year':
            return
        
        selected_season = int(self.season_dropdown.value)  # Get the selected season
        season_type_value = self.season_type_dropdown.value

        # Download data based on season and type
        if season_type_value == 'Regular Season':
            self.all_games_from_selection = self.downloader.download_regular_season(
                selected_season, 
                output_widget=self.loading_ouput)
        else:
            self.all_games_from_selection = self.downloader.download_playoffs(
                selected_season, 
                output_widget=self.loading_ouput)

        # Ensure that all_games is not empty before proceeding
        if len(self.all_games_from_selection) > 0:
            # Convert keys to a list and update the slider max value
            self.game_slider.max = len(self.all_games_from_selection) - 1
            self.update_game_plot(0)
            self.update_event_plot(0, 0)
        else:
            print("No games were found given RinkEventDebugTool's selection values!")


    # Update plot based on game ID and event ID
    def update_event_plot(self, game_index, event_index, home_abbrev=None, away_abbrev=None):
        if len(self.filtered_events) > 0 and event_index < len(self.filtered_events):
            event = self.filtered_events[event_index]
            x = event.get("details", {}).get("xCoord", None)
            y = event.get("details", {}).get("yCoord", None)
            if x is not None and y is not None:
                self.plot_event_on_rink(x, y, event, home_abbrev, away_abbrev)
            self.display_event_info(event)  # Display event info after the image


    # Update plot based on game ID and show game information
    def update_game_plot(self, game_index):
        game_ids = list(self.all_games_from_selection.keys())
        if game_index < len(game_ids):
            game_id = game_ids[game_index]
            game_data = self.all_games_from_selection.get(game_id, {})
            home_abbrev, away_abbrev = self.get_game_info(game_data)  # Update game info text
            
            # Filter events with typeDescKey "shot-on-goal" or "goal"
            self.filtered_events = [e for e in game_data.get("plays", []) if e.get("typeDescKey") in self.event_types]
            
            # Update the event slider max value to match the number of filtered events
            self.event_slider.max = len(self.filtered_events) - 1  # Update event slider based on filtered events
            # Reset the event_slider to 0
            self.event_slider.value = 0
            self.update_event_plot(game_index, 0, home_abbrev, away_abbrev)  # Update the plot based on the first event


    # Function to handle game id slider changes
    def on_game_change(self, change):
        game_index = change['new']
        self.update_game_plot(game_index)
        self.update_event_plot(0, 0)
        
        
    # Update plot based on event ID
    def on_event_change(self, change):
        game_index = self.game_slider.value
        event_index = change['new']
        # Get the current game data
        game_ids = list(self.all_games_from_selection.keys())
        if game_index < len(game_ids):
            game_id = game_ids[game_index]
            game_data = self.all_games_from_selection.get(game_id, {})
            home_abbrev = game_data.get("homeTeam", {}).get("abbrev", "Unknown Home Abbrev")
            away_abbrev = game_data.get("awayTeam", {}).get("abbrev", "Unknown Away Abbrev")
            self.update_event_plot(game_index, event_index, home_abbrev, away_abbrev)
        
        
    def display_layout(self):
        # Observe registrations
        self.event_slider.observe(self.on_event_change, names='value')    
        self.season_dropdown.observe(self.on_season_or_type_change, names='value')
        self.season_type_dropdown.observe(self.on_season_or_type_change, names='value')
        self.game_slider.observe(self.on_game_change, names='value')
        
        display(self.layout)
        