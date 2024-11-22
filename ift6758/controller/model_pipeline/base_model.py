from abc import abstractmethod
import datetime
import pickle

import pandas as pd
import sklearn.calibration
import matplotlib.pyplot as plt

import sklearn.metrics
import numpy as np

import os
import wandb

from ift6758.controller.model_pipeline.train_validation_sets_generator import TrainValidatorSetGenerator

class BaseModel:

    @staticmethod
    def init_plot_cache():
        """
        Crete an empty dictionnary to fill with plots axes.
        Useful to plot multiple line together at the end.
        Returns:
            the said dict
        """
        dict = {
            "roc":[],
            "goal_rate_by_percentile":[],
            "cumulative_goals_by_percentile":[],
            "reliability_diagram":[],
            "loss":[],
            "validation_error":[],
            "training_error":[],
        }
        return dict

    def __init__(self, dataframe: pd.DataFrame, column_y_name: str, validation_ratio: float =0.2, use_smote=False):
        """
        Initialize a base abstract model with it's data to use.
        provided dataframe must include, all x_train, y_train columns and rows.
        This class will handle the validation sets creation

        Args:
            use_smote: set if we should address imbalance in classes using smote
            validation_ratio: ratio of the dataframe that should be kept for validation. Default is 0.2
            dataframe: dataframe including all data (both x and y) and all none-test data
            column_y_name: string to specify the name of the column which represent our Y to predict
        """

        # Create id based on actual date time
        self.model_id = ""
        date_now = datetime.datetime.now()
        self.model_id += date_now.strftime("%m")
        self.model_id += '_'
        self.model_id += date_now.strftime("%y")
        self.model_id += '_'
        self.model_id += date_now.strftime("%d")
        self.model_id += '-'
        self.model_id += date_now.strftime("%H")
        self.model_id += date_now.strftime("%H")
        self.model_id += date_now.strftime("%S")

        # Create sets (shuffled and separated)
        setGenerator = TrainValidatorSetGenerator(dataframe)
        if use_smote:
            setGenerator.use_smote(column_y_name)

        training_set, validation_set = setGenerator.get_sets(validation_ratio)

        self.X_train = training_set.copy()
        self.X_train = self.X_train.drop(columns=[column_y_name])
        self.Y_train = training_set[column_y_name]

        self.X_validation = validation_set.copy()
        self.X_validation = self.X_validation.drop(columns=[column_y_name])
        self.Y_validation = validation_set[column_y_name]

        self.hyperparameters: dict = self.default_hyperparameters()
        self.model = None

        self.data_dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
        self.model_name = f"{self.__class__.__name__}_model_{self.model_id}"

        # Metrics to follow
        self.loss_curve = None
        self.training_error_curve = None
        self.validation_error_curve = None

        self.axes_dict = BaseModel.init_plot_cache()

    def set_custom_model_name(self, name):
        self.model_name = f"{name}_{self.model_id}"

    def __get_model_path(self) -> str:
        path = os.path.join(self.data_dir_path, "models")
        if not os.path.exists(path):
            os.makedirs(path)

        return os.path.join(path, f"{self.model_name}.pkl")

    def __get_plots_path(self, plot_name) -> str:
        path = os.path.join(self.data_dir_path, "plots", self.model_name)
        if not os.path.exists(path):
            os.makedirs(path)

        return os.path.join(path, plot_name)

    ##################################################################################
    # Abstract methods children needs to implement
    ##################################################################################
    @abstractmethod
    def default_hyperparameters(self) -> dict:
        pass
    @abstractmethod
    def get_confusion_matrix_class_names(self):
        pass
    @abstractmethod
    def train_model(self) :
        pass
    @abstractmethod
    def predict(self, x) -> list[int]:
        pass
    @abstractmethod
    def predict_proba(self, x) -> list[float]:
        pass

    def set_hyperparameter(self, hyperparam_name: str, hyperparam_value):
        if not self.hyperparameters.__contains__(hyperparam_name):
            print("Trying to set unknown hyper param")
            return

        if type(self.hyperparameters[hyperparam_name]) != type(hyperparam_value):
            print(f"Trying to set hyper param of type {type(self.hyperparameters[hyperparam_name])} with a value of type {type(hyperparam_value)}")
            return

        self.set_hyperparameters[hyperparam_name] = hyperparam_value

    ##################################################################################
    # Training metrics. We want to log those always
    ##################################################################################
    def _set_loss_curve(self, loss_curve: list[float]):
        self.loss_curve = loss_curve
    def _set_validation_error_curve(self, validation_error_curve: list[float]):
        self.training_error_curve = validation_error_curve
    def _set_training_error_curve(self, training_error_curve: list[float]):
        self.validation_error_curve = training_error_curve

    def save_model(self, model) :
        with open(self.__get_model_path(), 'wb') as file:
            pickle.dump(model, file)

    ##################################################################################
    ## Analysis plots. These can be done after training
    ## (these are requirement in milestone 2 questions)
    ##################################################################################
    def __plot_confusion_matrix(self, predictions):
        """
        Plot the confusion matrix of the model's predictions on validation set.
        Send it to WANDB
        """
        class_names = self.get_confusion_matrix_class_names()
        log_title = "Confusion Matrix"

        confusion_matrix = wandb.sklearn.plot_confusion_matrix(
            y_true=self.Y_validation,
            y_pred=predictions,
            labels=class_names)

        wandb.log({log_title: confusion_matrix})
        print("--Confusion Matrix sent to WANDB")

    def plot_combined_roc_curve(self, preds_proba_positive_class, y_truth):
        """
        Plot ROC curves.
        Save it locally and send it to WANDB

        Args:
            preds_proba_positive_class: 1d array of probabilities of positive class
            y_truth: 1d array of true class of each sample
        """
        # namings
        title = f"ROC Curve, {self.model_name}"
        save_path = self.__get_plots_path("ROC_Curve.png")

        # Plotting
        fig, ax = plt.subplots()
        plt.figure(figsize=(8, 6))

        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true=y_truth, y_score=preds_proba_positive_class)
        auc = sklearn.metrics.roc_auc_score(y_true=y_truth, y_score=preds_proba_positive_class)

        axe_label = f"{self.model_name} (AUC = {auc:.2f})"
        ax.plot(fpr, tpr, label=axe_label)

        ax.plot([0, 1], [0, 1], "k--", label="Random Classifier")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)

        fig.savefig(save_path)   # save the figure to file # do it before 'show' as the show function clear cache
        img = plt.imread(save_path)

        # Save axes for combined plot later
        # (X, Y, label)
        axe_tuple = (fpr, tpr, axe_label)
        self.axes_dict['roc'].append(axe_tuple)

        # plt.show() # if not showing, close it
        plt.close(fig)

        wandb.log({"ROC Curve": [wandb.Image(img, caption="ROC Curve")]})
        print("--ROC Curve sent to WANDB")

    def plot_goal_rate_by_percentile(self, preds_proba_positive_class, y_truth):
        """
        Plot the goal rate (goals / total shots) as a function of shot probability percentiles.
        Save it locally and send it to WANDB

        Args:
            preds_proba_positive_class: 1d array of probabilities of positive class
            y_truth: 1d array of true class of each sample
        """

        title = f"Goal Rate per percentile, {self.model_name}"
        save_path = self.__get_plots_path("ShotProb_Model_Percentile.png")

        fig, ax = plt.subplots()
        plt.figure(figsize=(8, 6))

        percentiles = np.percentile(preds_proba_positive_class, range(0, 101, 10))
        goal_rates = []

        for i in range(len(percentiles) - 1):
            mask = (preds_proba_positive_class >= percentiles[i]) & (preds_proba_positive_class < percentiles[i + 1])

            if mask.sum() > 0:
                goal_rate = y_truth[mask].mean() * 100  # Convert to percentage
            else:
                goal_rate = 0
            goal_rates.append(goal_rate)

        ax.invert_xaxis()  # Reverse x-axis direction
        # Reverse order for the x-axis
        ax.plot(range(101, 10, -10), list(reversed(goal_rates)), label=self.model_name)

        ax.set_xlabel("Shot Probability Model Percentile")
        ax.set_ylabel("Goals / (Shots + Goals) (%)")
        ax.set_ylim(0, 100)  # Set y-axis scale to 0–100
        ax.set_title(title)
        ax.grid()
        ax.legend()

        fig.savefig(save_path)   # save the figure to file # do it before 'show' as the show function clear cache
        img = plt.imread(save_path)

        # Save axes for combined plot later
        # (X, Y, label)
        axe_tuple = (range(101, 10, -10), list(reversed(goal_rates)), self.model_name)
        self.axes_dict['goal_rate_by_percentile'].append(axe_tuple)

        # plt.show() # if not showing, close it
        plt.close(fig)

        wandb.log({f"Goal Rate per percentile": [wandb.Image(img, caption="Goal Rate per percentile")]})
        print("--Goal Rate per percentile sent to WANDB")

    def plot_cumulative_goals_by_percentile(self, preds_proba_positive_class, y_truth):
        """
        Plot the cumulative percentage of goals as a function of shot probability percentiles for multiple models.
        Save it locally and send it to WANDB

        Args:
            preds_proba_positive_class: 1d array of probabilities of positive class
            y_truth: 1d array of true class of each sample
        """
        title = f"Cumulative % of Goals, {self.model_name}"
        save_path = self.__get_plots_path("cumul_goals_by_percentile.png")

        fig, ax = plt.subplots()
        plt.figure(figsize=(8, 6))

        # Sort probabilities and labels
        sorted_indices = np.argsort(preds_proba_positive_class)[::-1]
        sorted_labels = np.array(y_truth)[sorted_indices]

        # Compute cumulative goals
        cumulative_goals = np.cumsum(sorted_labels) / sorted_labels.sum() * 100  # Convert to percentage
        percentiles = np.linspace(10, 100, len(cumulative_goals))

        ax.invert_xaxis()  # Reverse x-axis direction
        ax.plot(percentiles, cumulative_goals, label=self.model_name)

        ax.set_xlabel("Shot Probability Model Percentile")
        ax.set_ylabel("Cumulative % of Goals")
        ax.set_ylim(0, 100)  # Set y-axis scale to 0–100
        ax.set_title(title)
        ax.grid()
        ax.legend()

        fig.savefig(save_path)  # save the figure to file # do it before 'show' as the show function clear cache
        img = plt.imread(save_path)

        # Save axes for combined plot later
        # (X, Y, label)
        axe_tuple = (percentiles, cumulative_goals, self.model_name)
        self.axes_dict['cumulative_goals_by_percentile'].append(axe_tuple)

        # plt.show() # if not showing, close it
        plt.close(fig)

        wandb.log(
            {f"Cumulative % of Goals": [wandb.Image(img, caption="Cumulative % of Goals")]})
        print("--Cumulative % of Goals sent to WANDB")

    def plot_reliability_diagram(self, preds_proba_positive_class, y_truth):
        """
        Plot the reliability diagram (calibration curve) for multiple models in one figure.
        Save it locally and send it to WANDB

        Args:
            preds_proba_positive_class: 1d array of probabilities of positive class
            y_truth: 1d array of true class of each sample
        """
        title = f"Reliability Diagram, {self.model_name}"
        save_path = self.__get_plots_path("reliability_diagram.png")

        fig, ax = plt.subplots()
        plt.figure(figsize=(8, 6))

        # Compute calibration curve using calibration_curve
        fraction_of_positives, mean_predicted_probabilities = sklearn.calibration.calibration_curve(
            y_truth, preds_proba_positive_class, n_bins=10, strategy="uniform"
        )
        # Add the calibration curve to the plot
        ax.plot(mean_predicted_probabilities, fraction_of_positives, marker="o", label=self.model_name)

        # Add perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")

        # Add plot details
        ax.set_title(title)
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.grid(alpha=0.3)

        fig.savefig(save_path)  # save the figure to file # do it before 'show' as the show function clear cache
        img = plt.imread(save_path)

        # Save axes for combined plot later
        # (X, Y, label)
        axe_tuple = (mean_predicted_probabilities, fraction_of_positives, self.model_name)
        self.axes_dict['reliability_diagram'].append(axe_tuple)

        # plt.show() # if not showing, close it
        plt.close(fig)

        wandb.log(
            {f"Reliability Diagram": [wandb.Image(img, caption="Reliability Diagram")]})
        print("--Reliability Diagram sent to WANDB")


    ##################################################################################
    # Init WANDB, train model, log metrics and log other plots analysis.
    ##################################################################################
    def __log_loss_curve(self):
        if self.loss_curve is None :
            print(f"Model {self.__class__.__name__} has no loss curve. Make sure this is want you want!")
            return

        # Log the loss curve (training loss at each iteration)
        for epoch, loss in enumerate(self.loss_curve):
            wandb.log({'epoch': epoch + 1, 'loss': loss})

        # Save axes for combined plot later
        # (loss_curve, label)
        axe_tuple = (self.loss_curve, self.model_name)
        self.axes_dict['loss'].append(axe_tuple)

    def __log_training_errors(self):
        if self.training_error_curve is None:
            print(f"Model {self.__class__.__name__} has no training errors. Make sure this is want you want!")
            return

            # Log the loss curve (training loss at each iteration)
        for epoch, training_error_rate in enumerate(self.training_error_curve):
            wandb.log({'epoch': epoch + 1, 'training_error_rate': training_error_rate})

        # Save axes for combined plot later
        # (loss_curve, label)
        axe_tuple = (self.training_error_curve, self.model_name)
        self.axes_dict['training_error'].append(axe_tuple)

    def __log_validation_error_curve(self):
        if self.validation_error_curve is None:
            print(f"Model {self.__class__.__name__} has no validation_error_curve. Make sure this is want you want!")
            return

            # Log the loss curve (training loss at each iteration)
        for epoch, validation_error_rate in enumerate(self.validation_error_curve):
            wandb.log({'epoch': epoch + 1, 'validation_error_rate': validation_error_rate})

        # Save axes for combined plot later
        # (loss_curve, label)
        axe_tuple = (self.validation_error_curve, self.model_name)
        self.axes_dict['validation_error'].append(axe_tuple)

    def __log_validation_final_accuracy(self, predictions):
        accuracy = sklearn.metrics.accuracy_score(self.Y_validation, predictions)
        wandb.log({'Validation Accuracy': accuracy})

    def evaluate_model(self):
        run_id = f"Evaluation_{self.model_name}"
        run = wandb.init(
            # Set the project where this run will be logged
            project="IFT6758.2024-A09",
            # Track hyperparameters and run metadata
            config=self.hyperparameters,
            id=run_id,
        )

        self.model = self.train_model()
        # Save model locally
        self.save_model(self.model)

        # Predict validaltion set
        predictions = self.predict(self.X_validation)
        preds_probabilities = self.predict_proba(self.X_validation)

        # Log training metrics
        self.__log_training_errors()
        self.__log_validation_error_curve()
        self.__log_loss_curve()
        self.__log_validation_final_accuracy(predictions)

        # Plot evaluation metrics
        self.plot_combined_roc_curve(preds_probabilities[:, 1], self.Y_validation)
        self.plot_goal_rate_by_percentile(preds_probabilities[:, 1], self.Y_validation)
        self.plot_cumulative_goals_by_percentile(preds_probabilities[:, 1], self.Y_validation)
        self.plot_reliability_diagram(preds_probabilities[:, 1], self.Y_validation)
        self.__plot_confusion_matrix(predictions)

        # Save model in wandb
        wandb.log_model(path=self.__get_model_path(), name=self.model_name)

        # Close current wandb run
        run.finish()

        return self.axes_dict
