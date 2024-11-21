
# import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE


class TrainValidatorSetGenerator:
    def __init__(self, data_set_df):
        self.data_set_df = data_set_df
        self.smote = False
        self.y_column_name = ''

    def use_smote(self, y_column_name: str):
        self.y_column_name = y_column_name
        self.smote = True

    def __shuffle_data(self):
        self.data_set_df = self.data_set_df.sample(frac=1).reset_index(drop=True)

    def __apply_smote(self):
        """
        Apply SMOTE to balance the training data.
        """
        if len(self.data_set_df[self.y_column_name].unique()) < 2:
            print("Skipping SMOTE: Training data contains only one class.")
            return

        smote = SMOTE(random_state=42)
        x_resampled, y_resampled = smote.fit_resample(self.data_set_df, self.data_set_df[self.y_column_name])

        # Update data
        # (we don't need y_resampled since it is also contained in x)
        self.data_set_df = x_resampled

    def get_sets(self, validation_ratio=0.2, shuffle=True):
        if shuffle:
            self.__shuffle_data()

        if self.smote:
            self.__apply_smote()

        validation_size = int( validation_ratio * len(self.data_set_df) )
        validation_set = self.data_set_df.iloc[0:validation_size]
        training_set = self.data_set_df.iloc[validation_size:]

        return training_set, validation_set
