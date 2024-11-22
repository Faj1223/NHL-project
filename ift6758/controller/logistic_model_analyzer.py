import pandas as pd
from setuptools.sandbox import save_path
from sklearn.calibration import CalibrationDisplay, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
import joblib
import datetime

import os
import wandb

from ift6758.controller.model_pipeline.base_model import BaseModel

class LogisticModelAnalyzer:
    def __init__(self, dataframe):
        """
        Initialize the class with the given dataframe.
        """
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

        self.dataframe = dataframe
        self.filtered_data = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.model = None
        self.data_dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "Q3")
        self.model_name = f"{self.__class__.__name__}_model_{self.model_id}"

    def __get_plots_path(self, plot_name) -> str:
        path = os.path.join(self.data_dir_path, "plots", self.model_name)
        if not os.path.exists(path):
            os.makedirs(path)

        return os.path.join(path, plot_name)

    def filter_data(self):
        """
        Filter out rows where shooter_id is 'Unknown'.
        """
        self.filtered_data = self.dataframe.dropna(subset=['shooting_distance','shot_angle', 'is_goal'])

    def validate_features(self, features):
        """
        Validate that all required features exist in the filtered dataset.

        Parameters:
        - features (list): List of features to validate.

        Raises:
        - ValueError if a feature is missing.
        """
        missing_features = [feature for feature in features if feature not in self.filtered_data.columns]
        if missing_features:
            raise ValueError(f"The following features are missing in the dataset: {missing_features}")

    def prepare_data(self, features, train_frac=0.8):
        """
        Split the data into training and validation sets based on selected features.

        Parameters:
        - features (list): List of features to include in the model.
        - train_frac (float): Fraction of data to use for training. Default is 0.8 (80%).
        """
        self.validate_features(features)

        print(f"Preparing data with features: {features}...")
        # Separate minority and majority classes
        minority_class = self.filtered_data[self.filtered_data['is_goal'] == 1]
        majority_class = self.filtered_data[self.filtered_data['is_goal'] == 0]

        # Perform stratified sampling for training and validation sets
        train_minority = minority_class.sample(frac=train_frac, random_state=42)
        train_majority = majority_class.sample(frac=train_frac, random_state=42)

        self.X_train = pd.concat([train_minority, train_majority])[features]
        self.y_train = pd.concat([train_minority, train_majority])['is_goal']

        val_minority = minority_class.drop(train_minority.index)
        val_majority = majority_class.drop(train_majority.index)

        self.X_val = pd.concat([val_minority, val_majority])[features]
        self.y_val = pd.concat([val_minority, val_majority])['is_goal']

        print("Data prepared:")
        print(f" - Training samples: {len(self.X_train)}")
        print(f" - Validation samples: {len(self.X_val)}")


    def apply_smote(self):
        """
        Apply SMOTE to balance the training data.
        """
        if len(self.y_train.unique()) < 2:
            print("Skipping SMOTE: Training data contains only one class.")
            return

        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(self.X_train, self.y_train)

        # Update training data
        self.X_train = pd.DataFrame(X_resampled, columns=self.X_train.columns)
        self.y_train = pd.Series(y_resampled)
        print(f"SMOTE applied. Training data balanced: {len(y_resampled)} samples.")

    def train_model(self):
        """
        Train a logistic regression model with default parameters.
        """
        self.model = LogisticRegression()
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        """
        Evaluate the model on the validation set and return the accuracy score.
        """
        predictions = self.model.predict(self.X_val)
        accuracy = accuracy_score(self.y_val, predictions)
        print(f"Validation Accuracy: {accuracy:.2f}")
        return accuracy, predictions

    def plot_confusion_matrix(self, predictions):
        """
        Plot the confusion matrix of the model's predictions.
        """
        # conf_matrix = confusion_matrix(self.y_val, predictions)
        # disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['No Goal', 'Goal'])
        # disp.plot(cmap='Blues')
        # plt.title("Confusion Matrix")
        # plt.show()

        run = wandb.init(
            # Set the project where this run will be logged
            project="IFT6758.2024-A09",
            # Track hyperparameters and run metadata
            config={
                "learning_rate": 0.01,
                "epochs": 10,
            },
        )

        # wandb.sklearn.plot_confusion_matrix(y_true, y_pred, labels)
        class_names = ["no_gaol", "goal"]
        confusion_matrix = wandb.sklearn.plot_confusion_matrix(
            self.y_val,
            predictions,
            class_names)

        wandb.log({"Q3-confusion_matrix": confusion_matrix})

        # Close current wandb run
        run.finish()

    def run_analysis(self,features=None, apply_smote=False):
        """
        Run the full analysis pipeline: filtering, oversampling, training, evaluation, and visualization.
        """
        if features is None:
            features = ["shooting_distance"]  # Default feature

        print("Filtering data...")
        self.filter_data()

        print("Preparing data...")
        self.prepare_data(features)

        if apply_smote:
            print("Applying SMOTE to oversample minority class...")
            self.apply_smote()

        print("Training model...")
        self.train_model()

        print("Evaluating model...")
        accuracy, predictions = self.evaluate_model()

        print("\nPlotting confusion matrix...")
        self.plot_confusion_matrix(predictions)

    def plot_combined_roc_curve(self, results, labels):
        """
        Plot ROC curves for multiple models on the same figure.
        """
        save_path = self.__get_plots_path("ROC_Curve.png")
        fig, ax = plt.subplots()

        plt.figure(figsize=(8, 6))
        for name, probabilities in results.items():
            fpr, tpr, _ = roc_curve(labels, probabilities)
            auc = roc_auc_score(labels, probabilities)
            ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")
        ax.plot([0, 1], [0, 1], "k--", label="Random Classifier")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve (All Models)")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        fig.savefig(save_path)  # save the figure to file # do it before 'show' as the show function clear cache
        img = plt.imread(save_path)
        # plt.show() # if not showing, close it
        plt.close(fig)

        wandb.log({"ROC Curve": [wandb.Image(img, caption="ROC Curve")]})
        print("--ROC Curve sent to WANDB")
        #plt.show()

    def plot_goal_rate_by_percentile(self, results, labels):
        """
        Plot the goal rate (goals / total shots) as a function of shot probability percentiles for multiple models.

        Parameters:
        - results (dict): Dictionary of model names and their predicted probabilities.
        - labels (array): True labels for the shots (1 for goal, 0 for no goal).
        """
        save_path = self.__get_plots_path("ShotProb_Model_Percentile.png")
        fig, ax = plt.subplots()

        plt.figure(figsize=(8, 6))

        for model_name, probabilities in results.items():
            percentiles = np.percentile(probabilities, range(0, 101, 10))
            goal_rates = []

            for i in range(len(percentiles) - 1):
                mask = (probabilities >= percentiles[i]) & (probabilities < percentiles[i + 1])
                if mask.sum() > 0:
                    goal_rate = labels[mask].mean() * 100  # Convert to percentage
                else:
                    goal_rate = 0
                goal_rates.append(goal_rate)

            # Reverse order for the x-axis
            ax.plot(range(101, 10, -10), goal_rates[::-1], label=model_name)

        ax.invert_xaxis()  # Reverse x-axis direction
        ax.set_xlabel("Shot Probability Model Percentile")
        ax.set_ylabel("Goals / (Shots + Goals) (%)")
        ax.set_ylim(0, 100)  # Set y-axis scale to 0–100
        ax.set_title("Goal Rate (All Models)")
        ax.grid()
        ax.legend()
        fig.savefig(save_path)  # save the figure to file # do it before 'show' as the show function clear cache
        img = plt.imread(save_path)
        # plt.show() # if not showing, close it
        plt.close(fig)

        wandb.log({"goal rate by percentile": [wandb.Image(img, caption="goal rate by percentile")]})
        print("--goal rate by percentile sent to WANDB")
        #plt.show()

    def plot_cumulative_goals_by_percentile(self, results, labels):
        """
        Plot the cumulative percentage of goals as a function of shot probability percentiles for multiple models.

        Parameters:
        - results (dict): Dictionary of model names and their predicted probabilities.
        - labels (array): True labels for the shots (1 for goal, 0 for no goal).
        """
        save_path = self.__get_plots_path("cumul_goals_by_percentile.png")
        fig, ax = plt.subplots()
        plt.figure(figsize=(8, 6))

        for model_name, probabilities in results.items():
            # Sort probabilities and labels
            sorted_indices = np.argsort(probabilities)[::-1]
            sorted_labels = np.array(labels)[sorted_indices]

            # Compute cumulative goals
            cumulative_goals = np.cumsum(sorted_labels) / sorted_labels.sum() * 100  # Convert to percentage
            percentiles = np.linspace(10, 100, len(cumulative_goals))

            ax.plot(percentiles, cumulative_goals, label=model_name)

        ax.invert_xaxis()  # Reverse x-axis direction
        ax.set_xlabel("Shot Probability Model Percentile")
        ax.set_ylabel("Cumulative % of Goals")
        ax.set_ylim(0, 100)  # Set y-axis scale to 0–100
        ax.set_title("Cumulative % of Goals (All Models)")
        ax.grid()
        ax.legend()
        fig.savefig(save_path)  # save the figure to file # do it before 'show' as the show function clear cache
        img = plt.imread(save_path)
        # plt.show() # if not showing, close it
        plt.close(fig)

        wandb.log({"cumulative goals by percentile": [wandb.Image(img, caption="cumulative goals by percentile")]})
        print("--cumulative goals by percentile sent to WANDB")
        #plt.show()

    def plot_reliability_diagram(self, results, labels):
        """
        Plot the reliability diagram (calibration curve) for multiple models in one figure.

        Parameters:
        - results (dict): Dictionary of model names and their predicted probabilities.
        - labels (array): True labels for the shots (1 for goal, 0 for no goal).
        """
        save_path = self.__get_plots_path("reliability_iagram.png")
        fig, ax = plt.subplots()
        plt.figure(figsize=(8, 6))

        for model_name, probabilities in results.items():
            # Compute calibration curve using calibration_curve
            fraction_of_positives, mean_predicted_probabilities = calibration_curve(
                labels, probabilities, n_bins=10, strategy="uniform"
            )
            # Add the calibration curve to the plot
            ax.plot(mean_predicted_probabilities, fraction_of_positives, marker="o", label=model_name)

        # Add perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")

        # Add plot details
        ax.set_title("Reliability Diagram (All Models)")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
        fig.savefig(save_path)  # save the figure to file # do it before 'show' as the show function clear cache
        img = plt.imread(save_path)
        # plt.show() # if not showing, close it
        plt.close(fig)

        wandb.log({"reliability diagram": [wandb.Image(img, caption="reliability diagram")]})
        print("--reliability diagram sent to WANDB")
        #plt.show()

    def evaluate_multiple_models(self):
        """
        Train and evaluate multiple models with different feature combinations and a random baseline.
        """
        run_id = f"Evaluation_{self.model_name}"
        run = wandb.init(project="IFT6758.2024-A09", job_type="model-evaluation",id=run_id)
        models = {
            "Distance Only": LogisticRegression(),
            "Angle Only": LogisticRegression(),
            "Distance and Angle": LogisticRegression(),
            "Random Baseline": None
        }

        features_dict = {
            "Distance Only": ["shooting_distance"],
            "Angle Only": ["shot_angle"],
            "Distance and Angle": ["shooting_distance", "shot_angle"],
            "Random Baseline": None
        }

        results = {}

        from sklearn.metrics import roc_auc_score, log_loss

        for name, model in models.items():

            run.tags = [name.replace(" ", "_").lower(), "logistic_regression", "evaluation"]

            if model:
                print(f"Training {name} model...")
                self.prepare_data(features_dict[name])  # Dynamically prepare data
                model.fit(self.X_train, self.y_train)
                probabilities = model.predict_proba(self.X_val)[:, 1]

                # Calculer et enregistrer les métriques
                auc = roc_auc_score(self.y_val, probabilities)
                logloss = log_loss(self.y_val, probabilities)
                print(f"Logging metrics: AUC={auc}, Log Loss={logloss}")
                run.log({"AUC": auc, "Log Loss": logloss})

                model_filename = f"{name.replace(' ', '_').lower()}_model.joblib"
                joblib.dump(model, model_filename)

                # Enregistrer le modèle dans W&B
                artifact = wandb.Artifact(name=f"{name.replace(' ', '_').lower()}_artifact", type="model")
                artifact.add_file(model_filename)
                run.log_artifact(artifact)
            else:
                print(f"Generating random probabilities for {name}...")
                probabilities = np.random.uniform(0, 1, len(self.y_val))

                # Calculer et enregistrer les métriques pour la baseline
                auc = roc_auc_score(self.y_val, probabilities)
                logloss = log_loss(self.y_val, probabilities)
                print(f"Logging metrics: AUC={auc}, Log Loss={logloss}")
                run.log({"AUC": auc, "Log Loss": logloss})

            results[name] = probabilities


        # Plot combined evaluation metrics
        self.plot_combined_roc_curve(results, self.y_val)
        self.plot_goal_rate_by_percentile(results, self.y_val)
        self.plot_cumulative_goals_by_percentile(results, self.y_val)
        self.plot_reliability_diagram(results, self.y_val)
        run.finish()