from sklearn.ensemble import IsolationForest
import numpy as np

def detect_anomaly(data):
    clf = IsolationForest(contamination=0.1)
    return clf.fit_predict(data)
