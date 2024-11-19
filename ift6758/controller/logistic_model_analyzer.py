import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np


class LogisticModelAnalyzer:
    def __init__(self, dataframe):
        """
        Initialize the class with the given dataframe.
        """
        self.dataframe = dataframe
        self.filtered_data = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.model = None

    def filter_data(self):
        """
        Filter out rows where shooter_id is 'Unknown'.
        """
        # self.filtered_data = self.dataframe[self.dataframe['shooter_id'] != "Unknown"]
        self.filtered_data = self.dataframe.dropna(subset=['distance_to_net', 'is_goal'])

    def prepare_data(self):
        """
        Prepare the data for training and validation by splitting it into train and validation sets.
        """
        # Split data manually
        minority_class = self.filtered_data[self.filtered_data['is_goal'] == 1]
        majority_class = self.filtered_data[self.filtered_data['is_goal'] == 0]

        # Ensure both classes are present in training
        train_minority = minority_class.sample(frac=0.8, random_state=42)
        train_majority = majority_class.sample(frac=0.8, random_state=42)

        # Combine and shuffle
        self.X_train = pd.concat([train_minority, train_majority])[['distance_to_net']]
        self.y_train = pd.concat([train_minority, train_majority])['is_goal']

        # Validation set
        val_minority = minority_class.drop(train_minority.index)
        val_majority = majority_class.drop(train_majority.index)
        self.X_val = pd.concat([val_minority, val_majority])[['distance_to_net']]
        self.y_val = pd.concat([val_minority, val_majority])['is_goal']


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
        conf_matrix = confusion_matrix(self.y_val, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['No Goal', 'Goal'])
        disp.plot(cmap='Blues')
        plt.title("Confusion Matrix")
        plt.show()

    def run_analysis(self,apply_smote=False):
        """
        Run the full analysis pipeline: filtering, oversampling, training, evaluation, and visualization.
        """
        print("Filtering data...")
        self.filter_data()

        print("Preparing data...")
        self.prepare_data()

        if apply_smote:
            print("Applying SMOTE to oversample minority class...")
            self.apply_smote()

        print("Training model...")
        self.train_model()

        print("Evaluating model...")
        accuracy, predictions = self.evaluate_model()

        print("\nPlotting confusion matrix...")
        self.plot_confusion_matrix(predictions)

    def plot_roc_curve(self, probabilities, labels):
        """
        Plot the ROC curve and calculate AUC.
        """
        fpr, tpr, _ = roc_curve(labels, probabilities)
        auc = roc_auc_score(labels, probabilities)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.show()

    def plot_goal_rate_by_percentile(self, probabilities, labels):
        """
        Plot the goal rate (goals / total shots) as a function of shot probability percentiles.
        """
        percentiles = np.percentile(probabilities, range(0, 101, 10))
        goal_rates = []

        for i in range(len(percentiles) - 1):
            mask = (probabilities >= percentiles[i]) & (probabilities < percentiles[i + 1])
            if mask.sum() > 0:
                goal_rate = labels[mask].mean() * 100  # Convert to percentage
            else:
                goal_rate = 0
            goal_rates.append(goal_rate)

        plt.figure(figsize=(8, 6))
        plt.plot(range(101, 10, -10), goal_rates[::-1], label="Model 1")  # Reverse order for the x-axis
        plt.gca().invert_xaxis()  # Reverse x-axis direction
        plt.xlabel("Shot probability model percentile")
        plt.ylabel("Goals / (Shots + Goals) (%)")  # Add (%) to the label
        plt.ylim(0, 100)  # Set y-axis scale to 0–100
        plt.title("Goal Rate")
        plt.grid()
        plt.legend()
        plt.show()

    def plot_cumulative_goals_by_percentile(self, probabilities, labels):
        """
        Plot the cumulative percentage of goals as a function of shot probability percentiles.
        """
        sorted_indices = np.argsort(probabilities)[::-1]
        sorted_labels = np.array(labels)[sorted_indices]

        cumulative_goals = np.cumsum(sorted_labels) / sorted_labels.sum() * 100  # Convert to percentage
        percentiles = np.linspace(10, 100, len(cumulative_goals))

        plt.figure(figsize=(8, 6))
        plt.plot(percentiles, cumulative_goals, label="Model 1")
        plt.gca().invert_xaxis()  # Reverse x-axis direction
        plt.xlabel("Shot probability model percentile")
        plt.ylabel("Cumulative % of Goals")  # Keep label as percentage
        plt.ylim(0, 100)  # Set y-axis scale to 0–100
        plt.title("Cumulative % of Goals")
        plt.grid()
        plt.legend()
        plt.show()

    def plot_reliability_diagram(self, probabilities, labels):
        """
        Plot the reliability diagram (calibration curve) using Scikit-learn's CalibrationDisplay.
        """
        # Use CalibrationDisplay to generate the reliability plot
        CalibrationDisplay.from_predictions(
            labels,
            probabilities,
            n_bins=10,
            strategy='uniform',
            name="Classifier"
        )

        # Add perfect calibration line
        plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")

        # Adjust the legend for clarity
        plt.legend(loc="best")
        plt.title("Reliability Diagram")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.grid(alpha=0.3)
        plt.show()

    def evaluate_probabilities(self):
        """
        Run all evaluations and produce the required plots.
        """
        print("Calculating predicted probabilities...")
        probabilities = self.model.predict_proba(self.X_val)[:, 1]
        print(self.model.classes_)  # Outputs: [0, 1]

        print("Plotting ROC curve...")
        self.plot_roc_curve(probabilities, self.y_val)

        print("Plotting goal rate by percentile...")
        self.plot_goal_rate_by_percentile(probabilities, self.y_val)

        print("Plotting cumulative goals by percentile...")
        self.plot_cumulative_goals_by_percentile(probabilities, self.y_val)

        print("Plotting reliability diagram...")
        self.plot_reliability_diagram(probabilities, self.y_val)