import numpy as np
import requests
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class ServingClient:
    def __init__(self, ip: str = "0.0.0.0", port: int = 5000, features=None):
        self.base_url = f"http://{ip}:{port}"
        logger.info(f"Initializing client; base URL: {self.base_url}")

        if features is None:
            features = ["distance"]
        self.features = features

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Formats the inputs into an appropriate payload for a POST request, and queries the
        prediction service. Retrieves the response from the server, and processes it back into a
        dataframe that corresponds index-wise to the input dataframe.

        Args:
            X (DataFrame): Input dataframe to submit to the prediction service.

        Returns:
            pd.DataFrame: Dataframe with predictions corresponding to the input.
        """
        try:
            url = f"{self.base_url}/predict"
            # Clean the DataFrame to replace NaN/Inf values
            X_cleaned = X.replace([np.inf, -np.inf], 0).fillna(0)

            # Log if there were any NaN/inf values in the input
            if X.isnull().values.any() or np.isinf(X.values).any():
                logger.warning("Input DataFrame contained NaN or inf values. Replaced them with 0.")

            # Convert DataFrame to JSON payload
            payload = X_cleaned.to_dict(orient="records")
            response = requests.post(url, json=payload)

            if response.status_code != 200:
                logger.error(f"Prediction request failed: {response.text}")
                response.raise_for_status()

            predictions = response.json().get("predictions", [])
            if not predictions:
                logger.warning("No predictions returned from server.")

            return pd.DataFrame({"predictions": predictions}, index=X.index)

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

    def logs(self) -> dict:
        """
        Retrieves server logs from the Flask app.

        Returns:
            dict: A dictionary containing the logs.
        """
        try:
            url = f"{self.base_url}/logs"
            response = requests.get(url)

            if response.status_code != 200:
                logger.error(f"Logs request failed: {response.text}")
                response.raise_for_status()

            return response.json()

        except Exception as e:
            logger.error(f"Error retrieving logs: {e}")
            raise

    def download_registry_model(self, workspace: str, model: str, version: str) -> dict:
        """
        Triggers a "model swap" in the service by requesting a model from the registry.

        Args:
            workspace (str): The workspace in WandB.
            model (str): The model name in the registry.
            version (str): The model version to download.

        Returns:
            dict: A response dictionary with the status and message.
        """
        try:
            url = f"{self.base_url}/download_registry_model"
            payload = {"workspace": workspace, "model": model, "version": version}
            response = requests.post(url, json=payload)

            if response.status_code != 200:
                logger.error(f"Model download request failed: {response.text}")
                response.raise_for_status()

            return response.json()

        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            raise
