from abc import abstractmethod

from sklearn.neural_network import MLPClassifier
from ift6758.controller.model_pipeline.base_model import BaseModel

class MLPClassifierModel(BaseModel):
    # Declare @abstractmethod
    def default_hyperparameters(self) -> dict:
        dict = {
            "solver":"sgd",
            "validation_fraction":0.1,
            "learning_rate":"constant",
            "learning_rate_init":0.001,
            "regularization_rate":0.001,
            "layers":(3,)
        }
        return dict

    # Declare @abstractmethod
    def get_confusion_matrix_class_names(self):
        class_names = ["no_goal", "goal"]
        return class_names

    # Declare @abstractmethod
    def train_model(self) :

        model = MLPClassifier(
            solver=self.hyperparameters["solver"],
            hidden_layer_sizes=self.hyperparameters["layers"],
            validation_fraction=self.hyperparameters["validation_fraction"],
            alpha=self.hyperparameters["regularization_rate"],
            learning_rate=self.hyperparameters["learning_rate"],
            learning_rate_init=self.hyperparameters["learning_rate_init"],
        )

        if self.hyperparameters["validation_fraction"] > 0.001:
            model.early_stopping = True

        model.fit(self.X_train, self.Y_train)

        self._set_loss_curve(model.loss_curve_)
        self._set_validation_error_curve(model.validation_scores_)

        return model

    # Declare @abstractmethod
    def predict(self, x) -> list[int]:
        return self.model.predict(x)

    # Declare @abstractmethod
    def predict_proba(self, x) -> list[float]:
        return self.model.predict_proba(x)


