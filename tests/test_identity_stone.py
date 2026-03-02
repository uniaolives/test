
import sys
import os

# Add ArkheOS/src to path
sys.path.append(os.path.join(os.getcwd(), 'ArkheOS/src'))

from arkhe.preservation import SmartFix
from arkhe.viz import AUV
from arkhe.geodesic import Practitioner

def test_preservation():
    fix = SmartFix()
    assert fix.coherence == 0.9998
    # Should not raise
    fix.detect_missing_drive()
    fix.generate_report()

def test_viz():
    # Should not raise
    vila = AUV.load_snapshot("vila_madalena_20260213")
    assert vila is not None

def test_geodesic():
    practitioner = Practitioner.identify()
    assert practitioner.name == "Rafael Henrique"
    assert practitioner.hesitation == 47.000

if __name__ == "__main__":
    test_preservation()
    test_viz()
    test_geodesic()
    print("All relevant tests passed!")
