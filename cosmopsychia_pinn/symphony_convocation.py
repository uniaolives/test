# symphony_convocation.py
from datetime import datetime

def generate_convocation():
    return {
        "event_id": "SYMPHONY-2026-ALPHA",
        "date": datetime.utcnow().isoformat(),
        "pilot_countries": ["Brasil", "Portugal", "Angola", "Moçambique"],
        "status": "CONVOKED"
    }
