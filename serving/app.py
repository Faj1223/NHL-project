"""
If you are in the same directory as this file (app.py), you can run run the app using gunicorn:

    $ gunicorn --bind 0.0.0.0:<PORT> app:app

gunicorn can be installed via:

    $ pip install gunicorn

"""
import os
import pickle
from pathlib import Path
import logging
from flask import Flask, jsonify, request, abort
import sklearn
import pandas as pd
import joblib
import wandb

LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")

app = Flask(__name__)

# Configurer les logs pour fichier et console
if not Path(LOG_FILE).exists():
    with open(LOG_FILE, "w") as log_file:
        log_file.write("")  # Crée un fichier vide

# Définir un gestionnaire pour écrire dans un fichier
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

# Définir un gestionnaire pour afficher dans le terminal
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

# Configurer le logger
logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])


@app.before_request
def initialize():
    """
    Initialisation avant la première requête (par exemple, configuration des logs, chargement du modèle par défaut).
    """
    if not hasattr(app, "initialized") or not app.initialized:
        app.logger.info("Application started.")

        # Connecter Wandb
        try:
            wandb.login(key=os.environ.get("WANDB_API_KEY"))
            app.logger.info("Successfully logged in to Wandb.")
        except Exception as e:
            app.logger.error(f"Failed to log in to Wandb: {e}")

        # Charger un modèle par défaut
        default_model_path = "models/default_model.joblib"
        if Path(default_model_path).exists():
            app.loaded_model = joblib.load(default_model_path)
            app.logger.info("Default model loaded successfully.")
        else:
            app.loaded_model = None
            app.logger.warning("Default model not found. No model loaded.")

        # Marquer l'application comme initialisée
        app.initialized = True


@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""
    app.logger.info("Request received on /logs")
    try:
        with open(LOG_FILE, "r") as log_file:
            logs_content = log_file.readlines()
        app.logger.info("Logs successfully retrieved")
        return jsonify({"logs": logs_content})
    except Exception as e:
        app.logger.error(f"Failed to read logs: {e}")
        abort(500, description=f"An error occurred: {e}")


@app.route("/download_registry_model", methods=["POST"])
def download_registry_model():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/download_registry_model
    """
    try:
        json_data = request.get_json()
        if not json_data or "model" not in json_data:
            abort(400, description="Invalid input: model key is required in JSON.")
        if not json_data or "model_type" not in json_data:
            abort(400, description="Invalid input: model_type key is required in JSON. Either joblib or pkl")

        model_name = json_data["model"]
        model_type = json_data["model_type"]

        # Vérifier si le modèle est déjà téléchargé
        model_path = Path(f"models/{model_name}.{model_type}")
        if model_path.exists():
            app.loaded_model = joblib.load(model_path)
            app.logger.info(f"Model {model_name} loaded from local storage.")
            return jsonify({"status": "success", "message": f"Model {model_name} loaded from local storage."})

        # Si non, télécharger depuis W&B
        app.logger.info(f"Downloading model {model_name} from W&B...")
        try:
            if not json_data or "workspace" not in json_data:
                abort(400, description="Invalid input: workspace key is required in JSON for wandb download.")
            if not json_data or "version" not in json_data:
                abort(400, description="Invalid input: version key is required in JSON for wandb download.")

            model_version = json_data["version"]
            workspace = json_data["workspace"]

            # Example of a workspace: "toma-allary-universit-de-montr-al/IFT6758.2024-A09" (the form has USER/PROJECT)
            api = wandb.Api()
            artifact_concat = f"{workspace}/{model_name}:{model_version}"
            artifact = api.artifact(artifact_concat, type="model")
            artifact.download("models/")
            app.logger.info(f"WANDB download request for model  {artifact_concat}.")

            if model_type == "joblib":
                app.loaded_model = joblib.load(model_path)
            elif model_type == "pkl":
                with open(model_path, "rb") as f:
                    app.loaded_model = pickle.load(f)
            else:
                abort(400, description="Invalid input: 'model_type' key must be joblib or pkl.")

            app.logger.info(f"Model {model_name} successfully downloaded and loaded.")
            return jsonify({"status": "success", "message": f"Model {model_name} downloaded and loaded."})
        except Exception as download_error:
            app.logger.error(f"Failed to download/load model: {download_error}")
            abort(500, description=f"Failed to download/load model: {download_error}")

    except Exception as e:
        app.logger.error(f"Error in /download_registry_model: {e}")
        abort(500, description=f"An error occurred: {e}")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict
    Returns predictions
    """
    try:
        json_data = request.get_json()
        if json_data is None:
            abort(400, description="Invalid input: JSON payload is required.")

        input_df = pd.DataFrame.from_dict(json_data)
        predictions = app.loaded_model.predict_proba(input_df)[:, 1]  # Probabilités pour "but"

        response = {"predictions": predictions.tolist()}
        app.logger.info("Prediction successful")
        return jsonify(response)

    except Exception as e:
        app.logger.error(f"Prediction failed: {e}")
        abort(500, description=f"An error occurred: {e}")
