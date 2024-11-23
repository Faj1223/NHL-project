import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import CalibrationDisplay
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score

class ProbabilisticModelVisualizer:
    def __init__(self, y_true: np.ndarray, y_probs: np.ndarray):
        """
        Initialise la classe avec les labels vrais et les probabilités prédites.
        
        :param y_true: Array-like, les labels vrais (0 ou 1).
        :param y_probs: Array-like, les probabilités prédites pour la classe 1.
        """
        self.y_true = np.array(y_true)
        self.y_probs = np.array(y_probs)

    def plot_roc_curve(self):
        """
        Plot the ROC curve and calculate AUC.
        """
        labels = self.y_true
        probabilities = self.y_probs
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

    def plot_goal_rate_by_percentile(self):
        """Trace le taux de but par centile de probabilité."""
        probabilities = self.y_probs
        labels = self.y_true
        
        percentiles = np.percentile(probabilities, np.arange(0, 101, 10))
        rates = []
        for i in range(len(percentiles) - 1):
            mask = (probabilities >= percentiles[i]) & (probabilities < percentiles[i + 1])
            rate = np.mean(labels[mask]) if np.sum(mask) > 0 else 0
            rates.append(rate)
        
        plt.figure(figsize=(8, 6))
        plt.plot(np.arange(100, 0, -10), np.array(rates) * 100, marker='o', linestyle='-', color='b')
        plt.xticks(np.arange(100, 0, -10))
        plt.yticks(np.arange(0, 110, 10))
        plt.xlabel("Centile de probabilité du modèle de tir")
        plt.ylabel("Taux de but (%)")
        plt.title("Taux de but par centile de probabilité")
        plt.grid(True)
        plt.gca().invert_xaxis()
        plt.show()

    def plot_cumulative_goals_by_percentile(self):
        """Trace le pourcentage cumulé de buts par centile de probabilité."""
        probabilities = self.y_probs
        labels = self.y_true

        sorted_indices = np.argsort(probabilities)[::-1]
        sorted_labels = labels[sorted_indices]

        cumulative_goals = np.cumsum(sorted_labels) / np.sum(sorted_labels)
        percentiles = np.arange(1, len(cumulative_goals) + 1) / len(cumulative_goals) * 100

        plt.figure(figsize=(8, 6))
        plt.plot(percentiles, cumulative_goals, label="Modèle")
        plt.xticks(np.arange(0, 110, 10))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xlabel("Centile de probabilité du modèle de tir")
        plt.ylabel("Pourcentage cumulé de buts")
        plt.title("Pourcentage cumulé de buts par centile de probabilité")
        plt.legend()
        plt.grid(True)
        plt.gca().invert_xaxis()
        plt.show()

    def plot_calibration_curve(self):
        """Trace le diagramme de calibration."""
        plt.figure(figsize=(10, 6))
        CalibrationDisplay.from_predictions(self.y_true, self.y_probs, n_bins=10)
        plt.title('Diagramme de fiabilité (Courbe de calibration)')
        plt.grid(True)
        plt.show()
