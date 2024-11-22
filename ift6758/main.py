
import sys

from ift6758.controller.model_pipeline.logistic_regression_model import LogisticRegressionModel
from ift6758.controller.model_pipeline.train_validation_sets_generator import TrainValidatorSetGenerator
from ift6758.controller.nhl_data_loader import NHLDataLoader

sys.path.append('../..')
import ift6758

from ift6758.controller.logistic_model_analyzer import LogisticModelAnalyzer

import importlib
importlib.reload(ift6758.controller.logistic_model_analyzer)
from ift6758.controller.logistic_model_analyzer import LogisticModelAnalyzer

import pandas as pd

import wandb
import os

#TODO: REMOVE this
def test_model_pipeline(df):
    df = df.dropna(subset=['shooting_distance', 'shot_angle', 'is_goal'])
    df = df[['shooting_distance', 'shot_angle', 'is_goal']]
    df['is_goal'] = df['is_goal'].astype(int)

    model = LogisticRegressionModel(df, 'is_goal', validation_ratio=0.1, use_smote=True)
    model.evaluate_model()

def main():

    # uncomment this only to set your local env var
    # DO NOT SUBMIT TO GIT REPO WITH YOUR SECRET KEY
    # os.environ['WANDB_API_KEY'] = 'your_key_here'


    my_key = os.environ.get('WANDB_API_KEY')
    wandb.login(key=my_key)

    loader = NHLDataLoader()
    df = loader.load_csv_files([2016, 2017, 2018, 2019, 2020])

    test_model_pipeline(df)

    #############################################################################
    # QUESTION 2
    #############################################################################

    # analyzer = LogisticModelAnalyzer(train_val_df)
    # analyzer.run_analysis()

    # todo ...

    #############################################################################
    # QUESTION ...
    #############################################################################




if __name__=="__main__":
    main()