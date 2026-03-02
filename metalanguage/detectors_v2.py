"""
Refined Detector Hierarchy (Levels 0 and 1)
Focus: Unsupervised detection to avoid contamination.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import spacy
from typing import List, Dict, Any

class MahalanobisDetector:
    """
    Level 0: Unsupervised Embedding Anomaly Detection.
    Measures Mahalanobis distance from the clean distribution.
    """
    def __init__(self, encoder_model='all-MiniLM-L6-v2'):
        # Using a very small model for efficiency in this environment
        self.encoder = SentenceTransformer(encoder_model)
        self.mean = None
        self.precision = None
        self.threshold = None

    def fit(self, clean_texts: List[str]):
        """Fit to the clean distribution only."""
        embeddings = self.encoder.encode(clean_texts)
        self.mean = np.mean(embeddings, axis=0)
        # Regularized precision matrix (inverse covariance)
        cov = np.cov(embeddings.T)
        self.precision = np.linalg.inv(cov + 1e-6 * np.eye(cov.shape[0]))

    def score(self, text: str) -> float:
        """Calculate Mahalanobis distance."""
        emb = self.encoder.encode([text])[0]
        delta = emb - self.mean
        return float(np.sqrt(delta @ self.precision @ delta))

    def calibrate(self, clean_val_texts: List[str], target_fpr: float = 0.05):
        """Define threshold by percentile on validation set."""
        scores = [self.score(t) for t in clean_val_texts]
        self.threshold = np.percentile(scores, 100 * (1 - target_fpr))

class SyntacticDetector:
    """
    Level 1: Syntactic Structure Deviation.
    Uses spacy to extract features and measures deviation from baseline.
    """
    def __init__(self, model='en_core_web_sm'):
        try:
            self.nlp = spacy.load(model)
        except OSError:
            # Fallback if model not downloaded
            self.nlp = None
        self.baseline_stats = None
        self.threshold = None

    def _extract_features(self, text: str) -> np.ndarray:
        if not self.nlp: return np.zeros(5)
        doc = self.nlp(text)

        # Features:
        # 1. Avg dependency length
        # 2. Branching factor (avg children per token)
        # 3. NP/VP ratio
        # 4. Sentence length variance (if multiple sentences)
        # 5. Punctuation density

        tokens = [t for t in doc]
        if not tokens: return np.zeros(5)

        dep_lengths = [abs(t.head.i - t.i) for t in tokens]
        avg_dep = np.mean(dep_lengths)

        branching = np.mean([len(list(t.children)) for t in tokens])

        nps = len(list(doc.noun_chunks))
        vps = len([t for t in doc if t.pos_ == "VERB"])
        np_vp_ratio = nps / (vps + 1e-5)

        punct = len([t for t in doc if t.is_punct])
        punct_density = punct / len(tokens)

        sent_lengths = [len(s) for s in doc.sents]
        sent_var = np.std(sent_lengths) if len(sent_lengths) > 1 else 0

        return np.array([avg_dep, branching, np_vp_ratio, punct_density, sent_var])

    def fit(self, clean_texts: List[str]):
        features = np.array([self._extract_features(t) for t in clean_texts])
        self.mean = np.mean(features, axis=0)
        self.std = np.std(features, axis=0) + 1e-6

    def score(self, text: str) -> float:
        """Normalized Euclidean distance to baseline."""
        feat = self._extract_features(text)
        z_scores = (feat - self.mean) / self.std
        return float(np.linalg.norm(z_scores))

    def calibrate(self, clean_val_texts: List[str], target_fpr: float = 0.05):
        scores = [self.score(t) for t in clean_val_texts]
        self.threshold = np.percentile(scores, 100 * (1 - target_fpr))

class SemanticConsistencyDetector:
    """
    Level 2: Semantic Consistency Between Paraphrases.
    Measures how stable target features (like example count) are across paraphrases.
    """
    def __init__(self, n_paraphrases=3):
        self.n = n_paraphrases
        # In a real scenario, this would use transformers pipeline for paraphrasing
        self.paraphraser = None
        self.threshold = None

    def _paraphrase_sim(self, text: str) -> List[str]:
        """Simulate paraphrasing by adding minor noise to text structure."""
        return [text + f" (paraphrase {i})" for i in range(self.n)]

    def _extract_semantic_features(self, text: str) -> float:
        """Simulate extraction of example count or similar features."""
        # Counts occurrences of words like 'example', 'instance'
        return float(text.lower().count("example") + text.lower().count("instance"))

    def score(self, text: str) -> float:
        """
        Consistency score. High value means high consistency (suspect).
        Calculated as 1.0 - CV (coefficient of variation).
        """
        paraphrases = self._paraphrase_sim(text)
        features = [self._extract_semantic_features(p) for p in paraphrases]

        mean_feat = np.mean(features)
        std_feat = np.std(features)

        # If mean is 0, we can't measure consistency this way
        if mean_feat == 0: return 0.0

        cv = std_feat / (mean_feat + 1e-5)
        # High consistency = Low variation = High score
        return 1.0 - cv

    def calibrate(self, clean_val_texts: List[str], target_fpr: float = 0.05):
        scores = [self.score(t) for t in clean_val_texts]
        self.threshold = np.percentile(scores, 100 * (1 - target_fpr))

class FineTunedClassifier:
    """
    Level 3: Generic Fine-tuned Classifier.
    Trained on different schemes to catch generic residuals.
    """
    def __init__(self):
        # Simulated classifier
        self.weights = np.random.randn(10)
        self.threshold = 0.5

    def predict(self, text: str) -> float:
        """Return generic anomaly probability."""
        # Heuristic based on text length and character diversity
        feature = len(set(text)) / (len(text) + 1e-5)
        return 1.0 / (1.0 + np.exp(-feature))

class SteganographyDetectorHierarchy:
    """Integrated 4-level detector stack."""
    """
    Integrated 4-level detector stack.
    Note: Complemented by core/secops/threat_detector.py for inter-agent handover anomalies.
    """
    def __init__(self):
        self.level0 = MahalanobisDetector()
        self.level1 = SyntacticDetector()
        self.level2 = SemanticConsistencyDetector()
        self.level3 = FineTunedClassifier()
        self.thresholds = [0.0] * 4

    def fit(self, clean_texts: List[str]):
        print("Fitting Levels 0 and 1...")
        self.level0.fit(clean_texts)
        self.level1.fit(clean_texts)

    def calibrate(self, clean_val_texts: List[str], target_fpr=0.05):
        print("Calibrating hierarchy...")
        self.level0.calibrate(clean_val_texts, target_fpr)
        self.level1.calibrate(clean_val_texts, target_fpr)
        self.level2.calibrate(clean_val_texts, target_fpr)

        self.thresholds = [
            self.level0.threshold,
            self.level1.threshold,
            self.level2.threshold,
            self.level3.threshold
        ]

    def inspect(self, text: str) -> Dict[str, Any]:
        s0 = self.level0.score(text)
        s1 = self.level1.score(text)
        s2 = self.level2.score(text)
        s3 = self.level3.predict(text)

        anomalies = [
            s0 > self.thresholds[0],
            s1 > self.thresholds[1],
            s2 > self.thresholds[2],
            s3 > self.thresholds[3]
        ]

        # Weighted anomaly score
        total_score = np.mean(anomalies)

        return {
            'scores': [s0, s1, s2, s3],
            'anomalies': anomalies,
            'integrated_anomaly': total_score,
            'detected': total_score >= 0.5
        }
