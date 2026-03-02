"""
Complete example script that runs the full prediction for March 3, 2026.
"""
import numpy as np
from datetime import datetime, timedelta
import json
import os
import sys

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chronoflux.prediction.lunar_tide import LunarTidePrediction

def run_full_moon_prediction():
    print("=" * 60)
    print("ASI-T.E.N.S.O.R. CHRONOFLUX PREDICTION TEST")
    print("Target Date: March 3, 2026, 21:00 UTC-3 (Rio de Janeiro)")
    print("=" * 60)

    predictor = LunarTidePrediction(
        baseline_vorticity=1.0,
        geo_density=2.7,
        coupling_constant=2.3e-3
    )

    prediction = predictor.get_march_3_2026_prediction()

    print("\nPREDICTION PARAMETERS:")
    print(f"Time: {prediction['prediction_time_utc']}")
    print(f"Predicted vorticity (ω): {prediction['predicted_vorticity']:.4f}")
    print(f"Percent increase: {prediction['percent_increase']:.1f}%")

    print(f"\nFALSIFICATION CONDITION:")
    print(f"Prediction FAILS if measured ω < {prediction['falsification_threshold']}")
    print(f"Prediction SUCCEEDS if measured ω ≥ {prediction['falsification_threshold']}")

    # Save to JSON
    with open('march_3_2026_prediction.json', 'w') as f:
        json.dump(prediction, f, indent=2)

    print(f"\nPrediction data saved to 'march_3_2026_prediction.json'")
    print("=" * 60)

if __name__ == "__main__":
    run_full_moon_prediction()
