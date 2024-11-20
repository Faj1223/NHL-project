
import sys
sys.path.append('../..')
import ift6758
import ift6758.controller.data_formatting_utils as data_formatting_utils

import importlib
importlib.reload(ift6758.controller.data_formatting_utils)
import ift6758.controller.data_formatting_utils as data_formatting_utils
import pandas as pd

from ift6758.controller.logistic_model_analyzer import LogisticModelAnalyzer

from ift6758.controller.nhl_data_downloader import NHLDataDownloader

import wandb
import os


def main():

    # uncomment this only to set your local env var
    # DO NOT SUBMIT TO GIT REPO WITH YOUR SECRET KEY
    # os.environ['WANDB_API_KEY'] = 'your_key_here'

    my_key = os.environ.get('WANDB_API_KEY')
    wandb.login(key=my_key)

    downloader = NHLDataDownloader()
    # range(2016, 2020) will download all seasons from 2016 to 2019
    train_val_df = downloader.load_season_data(season_range=range(2016, 2020))


    #############################################################################
    # QUESTION 2
    #############################################################################

    analyzer = LogisticModelAnalyzer(train_val_df)
    analyzer.run_analysis()

    # todo ...

    #############################################################################
    # QUESTION ...
    #############################################################################




if __name__=="__main__":
    main()



""" # Example of unused code: remove later
    formatter = data_formatting_utils.TableFormatter(2016, 2020)
    formatter.get_milestone_2_q2_formatting()

    formatter.formatted_df.to_csv("data/concat_CSV/milestone2_q2_concat_train.csv", index=False)

    formatter_2 = data_formatting_utils.TableFormatter(2021, 2021)
    # formatter_2.get_milestone_2_q2_formatting()

    formatter_2.formatted_df.to_csv("data/concat_CSV/milestone2_q2_concat_test.csv", index=False)
"""