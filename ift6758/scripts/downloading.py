import os
import sys

if os.path.join(os.path.dirname(os.path.dirname(__file__)), "controller") not in sys.path:
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "controller"))

from nhl_data_downloader import NHLDataDownloader

# À compléter (par exemple start_season = 2016)
start_season = 2016

# À compléter (par exemple end_season = 2016)
end_season = 2016

downloader = NHLDataDownloader()

downloader.download_all_seasons_play_by_play(start_season, end_season)