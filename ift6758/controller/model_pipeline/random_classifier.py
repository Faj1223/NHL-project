from abc import abstractmethod

import numpy as np

from ift6758.controller.model_pipeline.base_model import BaseModel

class RandomClassifierModel(BaseModel):
    # Declare @abstractmethod
    def default_hyperparameters(self) -> dict:
        dict = {
            "distribution":"uniform"
        }
        return dict

    # Declare @abstractmethod
    def get_confusion_matrix_class_names(self):
        class_names = ["no_goal", "goal"]
        return class_names

    # Declare @abstractmethod
    def train_model(self) :
        # no training to do
        return self

    # Declare @abstractmethod
    def predict(self, x) -> np.array:
        return (np.array(self.predict_proba(x)[:, 1]) > 0.5).astype(int).tolist()

    # Declare @abstractmethod
    def predict_proba(self, x) -> np.array:
        proba = np.zeros((len(x), 2))
        proba[:, 0] = np.random.uniform(0, 1, len(x)).tolist()
        proba[:, 1] = 1 - proba[:, 0]
        return proba


