import torch
import numpy as np
from datetime import datetime
from src.escudo.rio_social_monitor import RioSocialMonitor, SocialIndicator

def test_rio_social_monitor():
    print("Testing RioSocialMonitor...")
    monitor = RioSocialMonitor()

    indicator = SocialIndicator(
        timestamp=datetime.now(),
        protest_intensity=1.0,
        crime_rate_delta=0.0,
        economic_volatility=2.0,
        media_polarization=0.3,
        public_sentiment=0.5,
        institutional_trust=0.7,
        international_perception=0.8,
        cross_class_dialogue=0.6,
        inter_agency_cooperation=0.8,
        diplomatic_channel_openness=0.9
    )

    s_index = monitor.calculate_s_index(indicator)
    print(f"Calculated S-index: {s_index}")

    threat = monitor.classify_threat(s_index)
    print(f"Threat level: {threat}")

    trend = monitor.trend()
    print(f"Trend: {trend}")

    assert s_index > 0
    print("RioSocialMonitor test passed.")

if __name__ == "__main__":
    try:
        test_rio_social_monitor()
        print("\nProtocolo Escudo Python components verified successfully.")
    except Exception as e:
        print(f"\nVerification failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
