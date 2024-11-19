
import pandas as pd
import numpy as np


class TrainValidatorSetGenerator:
    def __init__(self, data_set_df):
        self.data_set_df = data_set_df

    def shuffle_data(self):
        self.data_set_df = self.data_set_df.sample(frac=1).reset_index(drop=True)

    def get_sets(self, validation_ratio=0.2, shuffle=True):
        if shuffle:
            self.shuffle_data()

        validation_size = int( validation_ratio * len(self.data_set_df) )
        validation_set = self.data_set_df.iloc[0:validation_size]
        training_set = self.data_set_df.iloc[validation_size:]

        return training_set, validation_set
