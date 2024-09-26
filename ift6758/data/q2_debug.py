import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import ipywidgets as widgets
from IPython.display import display, clear_output
import json

from ift6758.data import NHLDataDownloader

#%%
downloader = NHLDataDownloader()
# Global data storage
all_games = {}
game_id = ""
event_id = ""

# Widgets to select season type, game ID, and event ID
season_type = widgets.Dropdown(options=['Regular Season', 'Playoffs'], description='Season Type')
game_slider = widgets.IntSlider(min=0, description='Game ID')
event_slider = widgets.IntSlider(min=0, description='Event ID')

# Widgets to display game info and event info
game_info_text = widgets.Text(value="", description="Game Info", disabled=True, layout=widgets.Layout(width="90%"))
event_info_text = widgets.Text(value="", description="Event Info", disabled=True, layout=widgets.Layout(width="90%"))

# Widget to display the rink image
rink_image_output = widgets.Output()

# Function to plot event coordinates on the rink image
def plot_event_on_rink(x, y):
    rink_img_path = os.path.join('..','..','figures', 'nhl_rink.png')  # Update this to the correct path for the rink image
    rink_img = mpimg.imread(rink_img_path)

    with rink_image_output:
        rink_image_output.clear_output()  # Clear the previous plot
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(rink_img, extent=[-100, 100, -42.5, 42.5])
        if x is not None and y is not None:
            ax.scatter(x, y, color='blue', label="Event location")
            ax.legend()
        plt.show()


# Function to format and return game information
def get_game_info(game_data):
    game_start_time = game_data.get("startTimeUTC", "Unknown Date")
    home_abbrev = game_data.get("homeTeam", {}).get("abbrev", "Unknown Home Abbrev")
    away_abbrev = game_data.get("awayTeam", {}).get("abbrev", "Unknown Away Abbrev")
    home_goals = game_data.get("homeTeam", {}).get("score", 0)
    away_goals = game_data.get("awayTeam", {}).get("score", 0)
    home_sog = game_data.get("homeTeam", {}).get("sog", 0)
    away_sog = game_data.get("awayTeam", {}).get("sog", 0)

    # Format the game state (period, overtime, etc.)
    game_period = game_data.get("periodDescriptor", {}).get("periodType", "Unknown")

    # Return formatted game info similar to the image
    return (f"{game_start_time}\n"
            f"Game ID: {game_data.get('id', 'Unknown')}; {home_abbrev} (home) vs {away_abbrev} (away)\n\n"
            f"{game_period}\n"
            f"          Home       Away\n"
            f"Teams:    {home_abbrev}         {away_abbrev}\n"
            f"Goals:    {home_goals}           {away_goals}\n"
            f"SoG:      {home_sog}           {away_sog}\n")


# Function to format and return event details
def get_event_info(event):
    # Use json.dumps to pretty-print the event data
    event_info_json = json.dumps(event, indent=4)

    # Return the formatted JSON string
    return event_info_json

# Download data based on season type
def on_season_type_change(change):
    print(change)
    global all_games
    if change['new'] == 'Regular Season':
        all_games = downloader.download_regular_season(2016)  # Modify this as necessary
    else:
        all_games = downloader.download_playoffs(2016)  # Modify this as necessary
    # Ensure that all_games is not empty before proceeding
    if len(all_games) > 0:
        # Convert keys to a list and update the slider max value
        game_slider.max = len(all_games) - 1
        update_game_plot(0)
        update_event_plot(0, 0)

# Observe changes in season type dropdown
season_type.observe(on_season_type_change, names='value')


# Update plot based on game ID and event ID
def update_event_plot(game_index, event_index):
    global all_games
    global event_info_text

    # Get list of game IDs and access the selected game data
    game_ids = list(all_games.keys())
    if game_index < len(game_ids):
        game_id = game_ids[game_index]
        game_data = all_games.get(game_id, {})

        # Get the events for the selected game
        events = game_data.get("plays", [])
        if len(events) > 0 and event_index < len(events):
            event = events[event_index]
            x = event.get("details", {}).get("xCoord", None)
            y = event.get("details", {}).get("yCoord", None)
            if x is not None and y is not None:
                plot_event_on_rink(x, y)
            event_info_text.value = get_event_info(event)
        else:
            event_info_text.value = "No event data available."


# Update plot based on game ID and show game information
def update_game_plot(game_index):
    global all_games
    game_ids = list(all_games.keys())
    if game_index < len(game_ids):
        game_id = game_ids[game_index]
        game_data = all_games.get(game_id, {})
        clear_output()  # Clear the previous output to update new game info
        game_info_text.value = get_game_info(game_data)  # Update game info text
        event_slider.max = len(game_data.get("plays", [])) - 1  # Update event slider based on the number of events

def on_game_change(change):
    game_index = change['new']
    update_game_plot(game_index)
    update_event_plot(game_index, 0)

game_slider.observe(on_game_change, names='value')


# Update plot based on event ID
def on_event_change(change):
    game_index = game_slider.value
    event_index = change['new']
    update_event_plot(game_index, event_index)


event_slider.observe(on_event_change, names='value')

# Main function to execute when the script is run
def main():
    # Display widgets and text areas
    display(season_type, game_slider, game_info_text, event_slider, rink_image_output, event_info_text)

    # Initialize plot
    update_event_plot(0, 0)


# Check if the script is executed directly
if __name__ == "__main__":
    main()
    on_season_type_change({'new': 'Regular Season'})
    update_game_plot(2016020002)
    print(game_info_text.value)
    plot_event_on_rink(72, 2)
    update_event_plot(2016020002, 5)
    print(event_info_text.value)
