import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

class Plotation:
    def __init__(self, y_true, y_pred):
        """
        Initialize the class with true labels and predicted probabilities.
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)

    def plot_roc_curve(self):
        """Plot the ROC curve and calculate AUC."""
        fpr, tpr, _ = roc_curve(self.y_true, self.y_pred)
        auc = roc_auc_score(self.y_true, self.y_pred)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_goal_rate_by_percentile(self):
        """Plot goal rate by shot probability percentiles."""
        probabilities = self.y_pred
        labels = self.y_true
        percentiles = np.percentile(probabilities, np.arange(0, 101, 10))
        rates = []
        for i in range(len(percentiles) - 1):
            mask = (probabilities >= percentiles[i]) & (probabilities < percentiles[i + 1])
            rate = np.mean(labels[mask]) if np.sum(mask) > 0 else 0
            rates.append(rate)
        plt.figure()
        plt.plot(np.arange(10, 101, 10), np.array(rates) * 100, label="Goal Rate by Percentile")
        plt.title("Goal Rate by Percentile")
        plt.xlabel("Percentile of Predicted Probability")
        plt.ylabel("Goal Rate (%)")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_cumulative_goals_by_percentile(self):
        """Plot cumulative percentage of goals by shot probability percentiles."""
        probabilities = self.y_pred
        labels = self.y_true
        sorted_indices = np.argsort(probabilities)[::-1]
        sorted_labels = labels[sorted_indices]
        cumulative_goals = np.cumsum(sorted_labels) / np.sum(sorted_labels)
        percentiles = np.arange(1, len(cumulative_goals) + 1) / len(cumulative_goals) * 100
        plt.plot(percentiles, cumulative_goals, label="Cumulative Goals")
        plt.xlabel("Percentile of Predicted Probability")
        plt.ylabel("Cumulative Percentage of Goals")
        plt.title("Cumulative Goals by Percentile")
        plt.legend()
        plt.grid()
        plt.gca().invert_xaxis()
        plt.show()

    def plot_reliability_diagram(self):
        """Plot the reliability diagram (calibration curve)."""
        probabilities = self.y_pred
        labels = self.y_true
        prob_true, prob_pred = calibration_curve(labels, probabilities, n_bins=10)
        plt.figure()
        plt.plot(prob_pred, prob_true, label="Model Calibration")
        plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
        plt.title("Reliability Diagram")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.legend()
        plt.grid()
        plt.show()
