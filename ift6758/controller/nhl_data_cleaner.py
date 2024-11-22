import pandas as pd

class NHLDataCleaner():
    def __init__(self):
        return

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        columns = ["game_id", "period", "time_in_period", "event_type", "x_coord", "y_coord", "team_type"]
        df = df.dropna(subset=columns)
        df.reset_index(drop=True, inplace=True)
        return df
