import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

class XGBoostPipeline:
    def __init__(self, param_grid=None, random_state=42):
        """
        Initialise la classe avec une grille de paramètres et un état aléatoire pour la reproductibilité.
        """
        self.param_grid = param_grid if param_grid else {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 5, 7],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'gamma': [0, 1, 5],
            'min_child_weight': [1, 5, 10]
        }
        self.random_state = random_state
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            use_label_encoder=False,
            random_state=self.random_state
        )
        self.grid_search = None
        self.best_model = None

    def train(self, X_train, y_train):
        """
        Effectue la recherche par grille pour trouver les meilleurs hyperparamètres.
        """
        print("Recherche des meilleurs hyperparamètres...")
        self.grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=self.param_grid,
            scoring='roc_auc',
            cv=5,
            verbose=2,
            n_jobs=-1
        )
        self.grid_search.fit(X_train, y_train)
        self.best_model = self.grid_search.best_estimator_
        print("Meilleurs hyperparamètres trouvés :", self.grid_search.best_params_)

    def evaluate(self, X_test, y_test):
        """
        Évalue le modèle sur les données de test et affiche les métriques.
        """
        if not self.best_model:
            raise ValueError("Le modèle n'a pas encore été entraîné. Appelez 'train' d'abord.")

        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
        y_pred = self.best_model.predict(X_test)

        auc_score = roc_auc_score(y_test, y_pred_proba)
        print(f"AUC Score : {auc_score:.4f}")
        print("\nRapport de classification :\n", classification_report(y_test, y_pred))
        return auc_score

    def plot_roc_curve(self, X_test, y_test):
        """
        Trace la courbe ROC pour le modèle optimisé.
        """
        from sklearn.metrics import roc_curve
        if not self.best_model:
            raise ValueError("Le modèle n'a pas encore été entraîné. Appelez 'train' d'abord.")

        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)

        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.4f})", color='blue')
        plt.plot([0, 1], [0, 1], 'k--', color='gray', label='Classificateur Aléatoire')
        plt.xlabel('Taux de Faux Positifs (FPR)')
        plt.ylabel('Taux de Vrais Positifs (TPR)')
        plt.title('Courbe ROC')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()

    def plot_feature_importance(self):
        """
        Affiche l'importance des caractéristiques pour le modèle optimisé.
        """
        if not self.best_model:
            raise ValueError("Le modèle n'a pas encore été entraîné. Appelez 'train' d'abord.")

        plt.figure(figsize=(12, 6))
        xgb.plot_importance(self.best_model, importance_type='weight', ax=plt.gca())
        plt.title('Importance des caractéristiques')
        plt.show()
