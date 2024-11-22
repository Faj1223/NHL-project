import os
import sys

if os.path.join(os.path.dirname(os.path.dirname(__file__)), "controller") not in sys.path:
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "controller"))

from nhl_data_downloader import NHLDataDownloader

# À compléter
start_season = 2016

# À compléter
end_season = 2021

downloader = NHLDataDownloader()

downloader.download_all_seasons_play_by_play(start_season, end_season)
