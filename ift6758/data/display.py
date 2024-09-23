import requests
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

from ift6758.data import NHLDataDownloader

# Update the path to go two levels up, then into 'figures'
rink_img_path = os.path.join('..', '..', 'figures', 'nhl_rink.png')

# Load the rink image
rink_img = mpimg.imread(rink_img_path)

# Function to plot event coordinates on the rink image
def plot_event_on_rink(x, y):
    fig, ax = plt.subplots()
    ax.imshow(rink_img, extent=[-100, 100, -42.5, 42.5])
    ax.scatter(x, y, color='blue')
    plt.show()

# Update plot based on event ID
def update_plot(event_id):
    downloader = NHLDataDownloader()
    events = downloader.get_game_data('2016030245') # Example game ID
    event = events['liveData']['plays']['allPlays'][event_id]
    if 'coordinates' in event:
        x, y = event['coordinates']['x'], event['coordinates']['y']
        plot_event_on_rink(x, y)
    else:
        print("No coordinates available for this event.")

# Create widgets
event_slider = widgets.IntSlider(min=0, max=100, description='Event ID')
season_type = widgets.Dropdown(options=['Regular Season', 'Playoffs'], description='Season Type')
# Link Slider to the update function
def on_event_change(change):
    update_plot(change['new'])

event_slider.observe(on_event_change, names='value')
# Display widgets
display(event_slider, season_type)
