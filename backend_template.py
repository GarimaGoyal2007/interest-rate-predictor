"""Template for connecting your actual Colab model.

Rename this file to backend.py and implement run_model() using the real
functions from your completed notebook. Keep it in the same folder as app.py.
"""
import pandas as pd

def run_model(dataframe: pd.DataFrame, horizon: int, n_paths: int) -> dict:
    # 1. Call your real preprocessing/calibration functions.
    # 2. Generate forecasts for `horizon` days using `n_paths`.
    # 3. Calculate evaluation metrics on a valid test set.
    # 4. Return your actual results in this structure:
    #
    # return {
    #     "metrics": {"MAE": mae, "RMSE": rmse, "R2": r2},
    #     "predictions": predictions_dataframe,
    #     "simulation": simulation_dataframe
    # }
    raise NotImplementedError("Connect your actual Colab model functions here.")
