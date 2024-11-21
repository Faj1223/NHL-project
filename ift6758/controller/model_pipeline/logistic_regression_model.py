from abc import abstractmethod
from ift6758.controller.model_pipeline.base_model import BaseModel
from sklearn.linear_model import LogisticRegression

class LogisticRegressionModel(BaseModel):
    # Declare @abstractmethod
    def default_hyperparameters(self) -> dict:
        dict = {
            "solver":"lbfgs"
        }
        return dict

    # Declare @abstractmethod
    def get_confusion_matrix_class_names(self):
        class_names = ["no_goal", "goal"]
        return class_names

    # Declare @abstractmethod
    def train_model(self) :

        model = LogisticRegression()
        model.fit(self.X_train, self.Y_train)
        return model

    # Declare @abstractmethod
    def predict(self, x) -> list[int]:
        return self.model.predict(x)


