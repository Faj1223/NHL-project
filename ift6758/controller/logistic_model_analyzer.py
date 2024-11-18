import pandas as pd
from sklearn.calibration import calibration_curve
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

    def run_analysis(self):
        """
        Run the full analysis pipeline: filtering, oversampling, training, evaluation, and visualization.
        """
        print("Filtering data...")
        self.filter_data()

        print("Preparing data...")
        self.prepare_data()

        print("Applying SMOTE to oversample minority class...")
        self.apply_smote()

        print("Training model...")
        self.train_model()

        print("Evaluating model...")
        accuracy, predictions = self.evaluate_model()

        print("\nPlotting confusion matrix...")
        self.plot_confusion_matrix(predictions)

        print("\nPotential Observations:")
        print("- After oversampling, the dataset is balanced.")
        print("- This approach prevents the model from being biased toward the majority class.")

    def plot_roc_curve(self, probabilities, labels):
        """
        Plot the ROC curve and calculate AUC.
        """
        fpr, tpr, _ = roc_curve(labels, probabilities)
        auc = roc_auc_score(labels, probabilities)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.show()

    def plot_goal_rate_by_percentile(self, probabilities, labels):
        """
        Plot the goal rate (# goals / (# shots + # goals)) as a function of the shot probability percentiles.
        """
        percentiles = np.percentile(probabilities, np.arange(0, 101, 10))  # 10th, 20th, ..., 100th percentile
        rates = []
        for i in range(len(percentiles) - 1):
            mask = (probabilities >= percentiles[i]) & (probabilities < percentiles[i + 1])
            rate = np.mean(labels[mask]) if np.sum(mask) > 0 else 0
            rates.append(rate)

        plt.figure()
        plt.plot(np.arange(10, 101, 10), rates, label="Model 1")
        plt.title("Goal Rate")
        plt.xlabel("Shot Probability Model Percentile")
        plt.ylabel("Goals / (Shots + Goals)")
        plt.legend()
        plt.show()

    def plot_cumulative_goals_by_percentile(self, probabilities, labels):
        """
        Plot the cumulative percentage of goals as a function of the shot probability percentiles.
        """
        # Ensure labels and probabilities are aligned
        probabilities = pd.Series(probabilities).reset_index(drop=True)
        labels = pd.Series(labels).reset_index(drop=True)

        # Debugging step: Verify lengths
        if len(probabilities) != len(labels):
            raise ValueError(f"Mismatch in lengths: probabilities ({len(probabilities)}) and labels ({len(labels)})")

        # Sort probabilities and labels in descending order of probabilities
        sorted_indices = np.argsort(probabilities)[::-1]  # Descending order
        sorted_labels = labels.iloc[sorted_indices]  # Align labels with sorted probabilities

        # Compute cumulative sum of goals
        cumulative_goals = np.cumsum(sorted_labels) / np.sum(sorted_labels)  # Normalized cumulative sum
        percentiles = np.arange(1, len(cumulative_goals) + 1) / len(cumulative_goals) * 100

        # Plot the cumulative goals vs percentiles
        plt.plot(percentiles, cumulative_goals, label="Model")
        plt.xlabel("Shot probability model percentile")
        plt.ylabel("Cumulative % of goals")
        plt.title("Cumulative Percentage of Goals")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_reliability_diagram(self, probabilities, labels):
        """
        Plot the reliability diagram (calibration curve).
        """
        prob_true, prob_pred = calibration_curve(labels, probabilities, n_bins=10)
        plt.figure()
        plt.plot(prob_pred, prob_true, label="Model 1")
        plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
        plt.title("Reliability Diagram")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.legend()
        plt.show()

    def evaluate_probabilities(self):
        """
        Run all evaluations and produce the required plots.
        """
        print("Calculating predicted probabilities...")
        probabilities = self.model.predict_proba(self.X_val)[:, 1]  # Assuming class 1 corresponds to 'goal'

        print("Plotting ROC curve...")
        self.plot_roc_curve(probabilities, self.y_val)

        print("Plotting goal rate by percentile...")
        self.plot_goal_rate_by_percentile(probabilities, self.y_val)

        print("Plotting cumulative goals by percentile...")
        self.plot_cumulative_goals_by_percentile(probabilities, self.y_val)

        print("Plotting reliability diagram...")
        self.plot_reliability_diagram(probabilities, self.y_val)