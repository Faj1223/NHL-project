import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE


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

        # Print the original class distribution
        print("Original class distribution:")
        print(self.filtered_data['is_goal'].value_counts())
        print(f"Is Goal (True): {self.filtered_data['is_goal'].sum()}")  # Assuming True or 1 represents 'Goal'

        # Print the class distribution in the training set
        print("\nTraining set class distribution:")
        print(self.y_train.value_counts())
        print(f"Is Goal (True) in Training Set: {self.y_train.sum()}")

        # Print the class distribution in the validation set
        print("\nValidation set class distribution:")
        print(self.y_val.value_counts())
        print(f"Is Goal (True) in Validation Set: {self.y_val.sum()}")

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
