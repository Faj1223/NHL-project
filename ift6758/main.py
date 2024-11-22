import datetime
import sys
import time

import numpy as np

from ift6758.controller.model_pipeline.logistic_regression_model import LogisticRegressionModel
from ift6758.controller.model_pipeline.model_comparator import ModelComparator
from ift6758.controller.model_pipeline.random_classifier import RandomClassifierModel
from ift6758.controller.model_pipeline.train_validation_sets_generator import TrainValidatorSetGenerator
from ift6758.controller.model_pipeline.xgboost_model import XGBoostModel
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


def get_dist_angle_df(df):
    new_df = df.copy()
    new_df = new_df.dropna(subset=['shooting_distance', 'shot_angle', 'is_goal'])
    new_df = new_df[['shooting_distance', 'shot_angle', 'is_goal']]
    new_df['is_goal'] = new_df['is_goal'].astype(int)

    return new_df

def get_q4_features_df(df):
    """

    "time_in_period"                Secondes de jeu (Game seconds)
    "period"                        Période de jeu (Game period)
    "x_coord" "y_coord"             Coordonnées (x,y, colonnes séparées)
    "shooting_distance"             Distance de tir (Shot distance)
    "shot_angle"                    Angle de tir (Shot angle)
    "shot_type"                     Type de tir (Shot type)

    "last_event_type"               Dernier type d'événement (Last event type)
    "last_x_coord""last_y_coord"    Coordonnées du dernier événement (x, y, colonnes séparées)
    "time_since_last_event"         Temps écoulé depuis le dernier événement (secondes)
    "distance_from_last_event"      Distance depuis le dernier événement (Distance from the last event)

    "rebound"                       Rebond (bool) : Vrai si le dernier événement était aussi un tir, sinon False
    "rebound_angle"                 Changement d'angle de tir: Inclure seulement si le tir est un rebond, sinon 0.
    "speed_from_last_event"         «Vitesse» : distance de l'événement précédent / temps écoulé depuis l'ev precd.

    bonus here: not done yet
                                    temps écoulé depuis le début du jeu de puissance (secondes)
                                    Nombre de patineurs non-gardiens amicaux sur la glace
                                    Nombre de patineurs non-gardiens adverses sur la glace

    """

    new_df = df.copy()
    new_df = new_df.dropna(subset=['shooting_distance', 'shot_angle', 'is_goal'])
    new_df = new_df[
        ['time_in_period',
         'period',
         'x_coord',
         'y_coord',
         'shooting_distance',
         'shot_angle',
         'shot_type',
         'last_event_type',
         'last_x_coord',
         'last_y_coord',
         'time_since_last_event',
         'distance_from_last_event',
         'rebound',
         'rebound_angle',
         'speed_from_last_event',
         'is_goal']
    ]
    #true false
    new_df['is_goal'] = new_df['is_goal'].astype(int)
    new_df['rebound'] = new_df['rebound'].astype(int)

    new_df = new_df.replace(to_replace='unknown', value=np.nan)
    new_df= new_df.replace([np.inf, -np.inf], np.nan)

    new_df['last_x_coord'] = new_df['last_x_coord'].astype(float)
    new_df['last_y_coord'] = new_df['last_y_coord'].astype(float)

    # time
    new_df['time_in_period'] = new_df['time_in_period'].apply(lambda x: (time.strptime(x, '%M:%S').tm_min * 60) + time.strptime(x, '%M:%S').tm_sec)
    # new_df['time_since_last_event'] = new_df['time_since_last_event'].apply(lambda x: (time.strptime(x, '%M:%S').tm_min * 60) + time.strptime(x, '%M:%S').tm_sec)

    # types -> one hot encoding
    new_df = one_hot_encoding(new_df, 'shot_type')
    new_df = one_hot_encoding(new_df, 'last_event_type', prefix="last_event_is_")
    new_df = new_df.drop(columns=['shot_type', 'last_event_type'])

    new_df = new_df.fillna(value=0)
    return new_df

def one_hot_encoding(dataframe, column_name, prefix=""):
    categories = dataframe[column_name].unique()

    for category in categories:
        if category != 'unknown':
            dataframe[f"{prefix}{category}"] = (dataframe[column_name] == category).astype(int)
    return dataframe

#TODO: REMOVE this
def test_model_pipeline(df):
    # df = df.dropna(subset=['shooting_distance', 'shot_angle', 'is_goal'])
    # df = df[['shooting_distance', 'shot_angle', 'is_goal']]
    # df['is_goal'] = df['is_goal'].astype(int)

    model0 = XGBoostModel(df, 'is_goal', validation_ratio=0.1, use_smote=True)
    model0.set_hyperparameter("reg_alpha", float(0.1))
    model0.set_hyperparameter("learning_rate", float(1.0))
    model0.set_hyperparameter("n_estimators", int(100))


    model1 = LogisticRegressionModel(df, 'is_goal', validation_ratio=0.1, use_smote=True)
    model2 = LogisticRegressionModel(df, 'is_goal', validation_ratio=0.1, use_smote=False)
    model2.set_custom_model_name("No_smote_logreg")
    model3 = RandomClassifierModel(df, 'is_goal', validation_ratio=0.1, use_smote=True)

    comparator = ModelComparator([model0, model1])
    comparator.evaluate_models()
    comparator.plot_evaluation_together()

    # model.evaluate_model()

def test_hyper_param_configs_XGBoost(df):

    value_to_test = 10**(np.arange(-2,2,1.0))
    n_estimators_to_test = 10**(np.arange(2,4,1))
    model_list = []

    total = len(value_to_test)*len(value_to_test)#*len(n_estimators_to_test)

    for alpha in value_to_test:
        for learning_rate in value_to_test:
            # for n_est in n_estimators_to_test:
            m = XGBoostModel(df, 'is_goal', validation_ratio=0.1, use_smote=True)
            m.set_hyperparameter("reg_alpha", float(alpha))
            m.set_hyperparameter("learning_rate", float(learning_rate))
            m.set_hyperparameter("n_estimators", int(100))

            model_list.append(m)

    print(f"{total} models created for evaluation")


    comparator = ModelComparator(model_list)
    comparator.evaluate_models()
    comparator.plot_evaluation_together()

def main():

    # uncomment this only to set your local env var
    # DO NOT SUBMIT TO GIT REPO WITH YOUR SECRET KEY
    # os.environ['WANDB_API_KEY'] = 'your_key_here'


    my_key = os.environ.get('WANDB_API_KEY')
    wandb.login(key=my_key)

    loader = NHLDataLoader()
    df = loader.load_csv_files([2016, 2017, 2018, 2019, 2020])

    #############################################################################
    # QUESTION 3
    #############################################################################

    # analyzer = LogisticModelAnalyzer(df)
    # analyzer.run_analysis()

    #############################################################################
    # QUESTION 4
    #############################################################################

    # df = get_q4_features_df(df)

    #############################################################################
    # QUESTION 5
    #############################################################################
    # 5.1 (XGBoost on angle and distance only)
    # comment q4 for this
    # test_model_pipeline(df)

    #5.2 (XGBoost on Q4 features. Grid search for hyper Param)
    # test_hyper_param_configs_XGBoost(df)

    #5.3 (same as 5.2 with features selection)
    # optimal feature selection
    df_optimal = NHLDataLoader.get_optimal_features_dataframe(df, target_column='is_goal', k=5)
    test_model_pipeline(df_optimal)

    # todo ...

    #############################################################################
    # QUESTION ...
    #############################################################################




if __name__=="__main__":
    main()