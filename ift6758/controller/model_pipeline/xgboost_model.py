from abc import abstractmethod
from ift6758.controller.model_pipeline.base_model import BaseModel
import xgboost as xgb

class XGBoostModel(BaseModel):
    # Declare @abstractmethod
    def default_hyperparameters(self) -> dict:
        dict = {
            "early_stopping_rounds":2,
            "reg_alpha":0.01,
            "learning_rate": 0.01,
            "n_estimators": 100, #Number of gradient boosted trees
        }
        return dict

    # Declare @abstractmethod
    def get_confusion_matrix_class_names(self):
        class_names = ["no_goal", "goal"]
        return class_names

    # Declare @abstractmethod
    def train_model(self) :


        # Use "hist" for constructing the trees, with early stopping enabled.
        model = xgb.XGBClassifier(tree_method="hist",
                                  early_stopping_rounds=self.hyperparameters["early_stopping_rounds"],
                                  reg_alpha=self.hyperparameters["reg_alpha"],
                                  learning_rate=self.hyperparameters["learning_rate"],
                                  n_estimators=self.hyperparameters["n_estimators"],
                                  silent=True,
                                  eval_metric='logloss')
        # Fit the model, test sets are used for early stopping.
        model.fit(self.X_train, self.Y_train, eval_set=[(self.X_validation, self.Y_validation)])

        return model

    # Declare @abstractmethod
    def predict(self, x) -> list[int]:
        return self.model.predict(x)

    # Declare @abstractmethod
    def predict_proba(self, x) -> list[float]:
        return self.model.predict_proba(x)

