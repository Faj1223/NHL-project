import streamlit as st
import requests
import pandas as pd
from ift6758.client.serving_client import ServingClient

# Title of the App
st.title("Hockey Visualization App")

# Sidebar inputs for workspace, model, and version
st.sidebar.header("Workspace Configuration")
workspace = st.sidebar.text_input("Workspace", " ")
model = st.sidebar.text_input("Model", "")
version = st.sidebar.text_input("Version", "")

# Initialize serving_client in session state
if "serving_client" not in st.session_state:
    st.session_state.serving_client = ServingClient(ip="127.0.0.1", port=8000)

# Button to load model
if st.sidebar.button("Get model"):
    try:
        # Download the model from the registry
        result = st.session_state.serving_client.download_registry_model(workspace, model, version)
        if result.get("status") == "SUCCESS":
            st.sidebar.success(f"Model '{model}' (Version {version}) loaded successfully!")
        else:
            st.sidebar.error(f"Failed to load model: {result.get('message')}")
    except Exception as e:
        st.sidebar.error(f"Error downloading model: {e}")

# Main section
st.header("Game ID Input")
game_id = st.text_input("Enter Game ID", "2021020329")

# Ping game button
if st.button("Ping game"):
    try:
        endpoint = f"http://127.0.0.1:8000/predict"
        payload = {"game_id": game_id}

        # Send request to the prediction service
        response = requests.post(endpoint, json=payload)
        result = response.json()
        #
        # # Mocked API result if API isn't connected
        # if not result:
        #     result = {
        #         "teams": ["Canucks", "Avalanche"],
        #         "xG_actual": [3.2, 1.4],
        #         "score_actual": [3, 2],
        #         "events": [
        #             {"feature 1": 0.6321, "feature 2": 0.2581, "feature 3": 0.5314, "Model output": 0.4831},
        #             {"feature 1": 0.3004, "feature 2": 0.6790, "feature 3": 0.5043, "Model output": 0.7888},
        #             {"feature 1": 0.5291, "feature 2": 0.7031, "feature 3": 0.5849, "Model output": 0.8146}
        #         ]
        #     }

        # Display game results
        st.subheader(f"Game {game_id}: {result['teams'][0]} vs {result['teams'][1]}")
        st.write(f"**{result['teams'][0]} xG (actual):** {result['xG_actual'][0]} ({result['score_actual'][0]})")
        st.write(f"**{result['teams'][1]} xG (actual):** {result['xG_actual'][1]} ({result['score_actual'][1]})")

        # Display event predictions as a dataframe
        st.subheader("Data used for predictions (and predictions)")
        events_df = pd.DataFrame(result["events"])
        st.dataframe(events_df)

    except Exception as e:
        st.error(f"Error occurred: {e}")


