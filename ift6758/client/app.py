import streamlit as st
import requests
import pandas as pd

from game_client import GameClient
from serving_client import ServingClient

# Title of the App
st.title("Hockey Visualization App")

# Sidebar inputs for workspace, model, and version
st.sidebar.header("Workspace Configuration")
workspace = st.sidebar.text_input("Workspace", "toma-allary-universit-de-montr-al/IFT6758.2024-A09")
model = st.sidebar.text_input("Model", "distance_and_angle_model")
version = st.sidebar.text_input("Version", "latest")

# Initialize serving_client in session state
if "serving_client" not in st.session_state:
    st.session_state.serving_client = ServingClient(ip="0.0.0.0", port=5000)

# Button to load model
if st.sidebar.button("Get model"):
    try:
        # Download the model from the registry
        result = st.session_state.serving_client.download_registry_model(workspace, model, version)
        if result.get("status") == "success":
            st.sidebar.success(f"Model '{model}' (Version {version}) loaded successfully!")
        else:
            st.sidebar.error(f"Failed to load model: {result.get('message')}")
    except Exception as e:
        st.sidebar.error(f"Error downloading model: {e}")

# Maintain state for home and away team names
if "home_team_name" not in st.session_state:
    st.session_state.home_team_name = "Unknown Home Team"
if "away_team_name" not in st.session_state:
    st.session_state.away_team_name = "Unknown Away Team"

# Main section
st.header("Game ID Input")
game_id = st.text_input("Enter Game ID", "2021020329")

if st.button("Ping game"):
    # Initialize GameClient instance
    client = GameClient(game_id=game_id, prediction_url="http://0.0.0.0:5000/predict")
    processed_events = client.process_game()

    if not processed_events.empty:
        # # Check and update home and away team names from processed events
        # if not processed_events.loc[processed_events['team_type'] == 'home'].empty:
        #     st.session_state.home_team_name = processed_events.loc[processed_events['team_type'] == 'home', 'team_name'].iloc[0]
        # if not processed_events.loc[processed_events['team_type'] == 'away'].empty:
        #     st.session_state.away_team_name = processed_events.loc[processed_events['team_type'] == 'away', 'team_name'].iloc[0]

        st.session_state.home_team_name = client.home_team_name
        st.session_state.away_team_name = client.away_team_name

        # Calculate cumulative xG for home and away teams
        team_xg = client.calculate_team_xg(processed_events)

        # Calculate current scores
        home_score = processed_events[(processed_events['team_type'] == 'home') & (processed_events['is_goal'])].shape[0]
        away_score = processed_events[(processed_events['team_type'] == 'away') & (processed_events['is_goal'])].shape[0]

        # Calculate xG differences
        home_xg_diff = round(team_xg['home_xG'] - home_score, 2)
        away_xg_diff = round(team_xg['away_xG'] - away_score, 2)

        # Format xG values to 2 decimal places
        home_xg_formatted = f"{team_xg['home_xG']:.2f}"
        away_xg_formatted = f"{team_xg['away_xG']:.2f}"

        # Display game header with period, time left, and defending side
        period = processed_events['period'].iloc[0]
        time_left = processed_events['time_remaining_in_period'].iloc[0]
        home_defending_side = processed_events['home_team_defending_side'].iloc[
            0] if 'home_team_defending_side' in processed_events.columns else "Unknown"
        st.subheader(f"Game {game_id}: {st.session_state.home_team_name} vs {st.session_state.away_team_name}")
        st.write(f"Period {period} - {time_left} - {home_defending_side} ")

        # Display metrics for xG and current scores
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label=f"{st.session_state.home_team_name} xG (actual)", value=f"{home_xg_formatted} ({home_score})", delta=home_xg_diff)
        with col2:
            st.metric(label=f"{st.session_state.away_team_name} xG (actual)", value=f"{away_xg_formatted} ({away_score})", delta=away_xg_diff)

        # Display event predictions
        st.subheader("Data used for predictions (and predictions)")
        st.dataframe(processed_events)
    else:
        st.error("No new events to process or data returned from the prediction service.")
