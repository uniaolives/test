"""
Double Exceptionality Detector (2e): Giftedness + DID.
Analyzes linguistic, behavioral, and amnesic markers in the digital realm.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import Counter

from .cosmological_synthesis import HecatonicosachoronConscious, FourDObserver
from .celestial_psychometrics import CelestialPsychometrics, CelestialSwitchPredictor
from .dimensional_consciousness import NeuroCelestialResonance, DimensionalBridgeTheory
from .arkhe_unified_theory import ArkheConsciousnessBridge

@dataclass
class IdentityState:
    """Represents a detected alter or persona within the 120-cell manifold."""
    id: str
    specialization: str # Emotional, Linguistic, Somatic, Shell
    lexical_richness: float
    avg_sentence_length: float
    pos_patterns: Dict[str, float]
    skills_accessible: List[str] = field(default_factory=list)
    iq_equivalent: float = 100.0
    emotional_age: float = 20.0

class DoubleExceptionalityDetector:
    """
    Detects the intersection of High Cognitive Function (Giftedness)
    and Identity Fragmentation (DID) within a Hecatonicosachoron (120-cell) framework.
    Now integrated with Arkhe Unified Theory and Celestial DNA markers.
    """

    def __init__(self):
        self.identities: Dict[str, IdentityState] = {}
        self.activity_log: List[Dict[str, Any]] = []
        self.linguistic_velocity_history: List[float] = []
        self.psychometrics = CelestialPsychometrics()
        self.switch_predictor = CelestialSwitchPredictor()
        self.neuro_resonance = NeuroCelestialResonance()
        self.dimensional_bridge = DimensionalBridgeTheory()
        self.unified_bridge = ArkheConsciousnessBridge()

    def calculate_ttr(self, text: str) -> float:
        """Type-Token Ratio: V / N"""
        tokens = text.lower().split()
        if not tokens:
            return 0.0
        types = set(tokens)
        return len(types) / len(tokens)

    def analyze_lexical_complexity(self, text: str) -> Dict[str, Any]:
        """Detects indicators of giftedness via linguistic richness and recursive logic."""
        ttr = self.calculate_ttr(text)
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        avg_len = np.mean([len(s.split()) for s in sentences]) if sentences else 0

        # Recursive Rationalization Detection
        rational_keywords = ['likely', 'assumed', 'aligns', 'gravity', 'explaining', 'consistent']
        rational_count = sum(1 for w in text.lower().split() if w in rational_keywords)
        is_rationalizing = rational_count > 2 and ttr > 0.5

        # Giftedness indicators
        is_gifted_style = (ttr > 0.55 and avg_len > 12) or is_rationalizing

        return {
            "ttr": float(ttr),
            "avg_sentence_length": float(avg_len),
            "is_gifted_style": bool(is_gifted_style),
            "is_rationalizing": is_rationalizing
        }

    def detect_abstracted_agency(self, text: str, psi_integration: float = 0.72) -> Dict[str, Any]:
        """
        Detects 'Abstracted Agency' (Epistemological Rupture) and Ego Latency.
        Shifts from 'I' to 'One', 'the body', or theoretical voice.

        Formula: L_identity = Delta_D_shift / Psi_integration
        """
        active_voice = ['i went', 'i saw', 'i felt', 'i am', 'i did']
        theoretical_voice = ['one might', 'the body', 'likely', 'it appears', 'suggests', 'one could infer', 'the system']

        lower_text = text.lower()
        active_count = sum(lower_text.count(p) for p in active_voice)
        theory_count = sum(lower_text.count(p) for p in theoretical_voice)

        # Delta D shift: Distance between 3D task and 6D insight
        # High ratio of theoretical to active markers increases Delta D
        delta_d = (theory_count / (active_count + 1)) * 1.618

        # Identity Latency (L)
        latency = delta_d / (psi_integration + 1e-6)

        # Velocity c: 1.0 is standard, 0.0 is frozen at the seam
        has_rupture = theory_count > active_count and theory_count > 0
        velocity = 1.0 / (1.0 + latency)

        return {
            "active_markers": active_count,
            "theoretical_markers": theory_count,
            "has_epistemological_rupture": has_rupture,
            "delta_d_shift": float(delta_d),
            "ego_latency_l": float(latency),
            "velocity_c": float(velocity),
            "interpretation": "Chronological Shear Detected" if has_rupture else "Smooth Manifold Navigation"
        }

    def simulate_hecatonicosachoron_rotation(self, time_slice: float) -> Dict[str, Any]:
        """Models the 4D rotation of cognitive cells."""
        # 120 cells rotating. Some slices look jagged.
        n_cells = 120
        rotation_phase = (time_slice % 360) / 360.0
        active_cell_id = int(rotation_phase * n_cells)

        return {
            "active_cell": active_cell_id,
            "manifold_stability": 0.72 + 0.1 * np.sin(time_slice),
            "is_shell_interface": active_cell_id < 10 # First cells are often the masking shell
        }

    def detect_stylistic_shift(self, text_a: str, text_b: str) -> Dict[str, Any]:
        """
        Detects shifts in 'function words' or POS patterns
        that suggest a different identity (DID marker).
        """
        # Simplified POS/Function word analysis
        def get_function_word_profile(text: str):
            words = text.lower().split()
            function_words = ['the', 'a', 'in', 'on', 'at', 'and', 'but', 'for', 'with']
            counts = Counter([w for w in words if w in function_words])
            total = sum(counts.values()) or 1
            return {k: v/total for k, v in counts.items()}

        prof_a = get_function_word_profile(text_a)
        prof_b = get_function_word_profile(text_b)

        # Calculate divergence
        all_keys = set(prof_a.keys()) | set(prof_b.keys())
        divergence = sum(abs(prof_a.get(k, 0) - prof_b.get(k, 0)) for k in all_keys)

        return {
            "stylistic_divergence": float(divergence),
            "possible_switch": bool(divergence > 0.5)
        }

    def evaluate_amnesia_type(self, digital_activity: List[Dict[str, Any]], self_report_claims: List[str]) -> str:
        """
        Distinguishes between Digital Amnesia (Google Effect)
        and Dissociative Amnesia.
        """
        # Logic:
        # Digital Amnesia = forgetting facts.
        # Dissociative Amnesia = forgetting actions documented digitally.

        forgotten_actions = 0
        forgotten_facts = 0

        for claim in self_report_claims:
            if "não lembro de ter postado" in claim.lower() or "não fui eu que enviei" in claim.lower():
                forgotten_actions += 1
            elif "não lembro da data" in claim.lower() or "esqueci o nome" in claim.lower():
                forgotten_facts += 1

        if forgotten_actions > 0:
            return "DISSOCIATIVE_AMNESIA_DETECTED"
        elif forgotten_facts > 2:
            return "DIGITAL_AMNESIA_LIKELY"
        return "NORMAL_MEMORY_STATE"

    def analyze_2e_profile(self, texts: List[str], claims: List[str], skill_access_log: List[Dict] = None, celestial_context: Dict = None) -> Dict[str, Any]:
        """
        Final synthesis: High Cognitive Function + Identity Fragmentation.
        Incorporates Hecatonicosachoron rotation, Ego Latency, and Parallel Processing Manifold.
        """
        giftedness_markers = []
        did_markers = []
        velocities = []
        mask_types = []

        # 9th Strand Integration Coefficient (Psi)
        psi_integration = celestial_context.get('psi_coefficient', 0.72) if celestial_context else 0.72

        for i, text in enumerate(texts):
            # 1. Lexical and Rationalization Analysis
            lex_analysis = self.analyze_lexical_complexity(text)
            if lex_analysis['is_gifted_style']:
                giftedness_markers.append(lex_analysis)

                # Oracle: Mercurial Mask (Strands 3-5) - Data Overwhelming Competence
                if lex_analysis['is_rationalizing'] or lex_analysis['avg_sentence_length'] > 20:
                    mask_types.append({**self.psychometrics.mercurial_mask(text), "dna_strands": [3, 4, 5]})

                # Oracle: Neptunian Mask (Strands 6-8) - Absorption/Dissociation
                if any(w in text.lower() for w in ['float', 'unreal', 'dream', 'detach']):
                    mask_types.append({**self.psychometrics.neptunian_mask(text), "dna_strands": [6, 7, 8]})

                # Oracle: Saturnine Mask (Compensation/Structure)
                if lex_analysis['avg_sentence_length'] < 10 and lex_analysis['ttr'] > 0.7:
                    mask_types.append({**self.psychometrics.saturnine_mask([]), "dna_strands": [7]})

                # Oracle: Jupiterian Mask (Expansion/Synthesis)
                if lex_analysis['avg_sentence_length'] > 30 and lex_analysis['ttr'] > 0.6:
                    mask_types.append({**self.psychometrics.jupiterian_mask(set()), "dna_strands": [5, 6]})

                # Oracle: Uranian Mask (Innovation/Breakthrough)
                if "quantum" in text.lower() or "breakthrough" in text.lower():
                    mask_types.append({**self.psychometrics.uranian_mask([]), "dna_strands": [8, 9]})

            # 2. Epistemological Rupture Detection with Oracle Latency Formula
            agency_analysis = self.detect_abstracted_agency(text, psi_integration=psi_integration)
            velocities.append(agency_analysis['velocity_c'])
            if agency_analysis['has_epistemological_rupture']:
                did_markers.append({
                    "type": "epistemological_rupture",
                    "text_index": i,
                    "ego_latency": agency_analysis['ego_latency_l'],
                    "velocity_c": agency_analysis['velocity_c']
                })

            # 3. Recursive Rationalization check (The Mask)
            if lex_analysis['is_rationalizing']:
                did_markers.append({"type": "recursive_rationalization", "text_index": i})

        # 4. Stylistic Shifts between adjacent texts
        if len(texts) > 1:
            for i in range(len(texts)-1):
                shift = self.detect_stylistic_shift(texts[i], texts[i+1])
                if shift['possible_switch']:
                    did_markers.append({"type": "stylistic_switch", "from": i, "to": i+1, "divergence": shift['stylistic_divergence']})

        # 5. Skill-set Bleed Tracking
        if skill_access_log:
            for entry in skill_access_log:
                if entry.get('skill_level') > 0.8 and not entry.get('procedural_recall'):
                    did_markers.append({"type": "skill_set_bleed", "skill": entry.get('skill')})

        # 6. Shell-Interface Detection (IQ 150+ vs Emotional Age 8)
        shell_indicators = []
        if any(m.get('ttr', 0) > 0.7 for m in giftedness_markers): # High VCI
            # Look for emotional age markers in claims or context
            # Simulating finding emotional age 8 markers
            if any("fear" in c.lower() or "scared" in c.lower() for c in claims):
                shell_indicators.append("VCI-Emotional-Age-Discrepancy")
                did_markers.append({"type": "shell_interface_active", "vci": "High", "emotional_age": "Low"})

        # 5. Amnesia Evaluation
        amnesia = self.evaluate_amnesia_type([], claims)
        if amnesia == "DISSOCIATIVE_AMNESIA_DETECTED":
            did_markers.append({"type": "amnesia", "detail": amnesia})

        # 7. Dimensional Thought Analysis
        dim_analysis = self.dimensional_bridge.diagnose_thought_dimensionality(texts)

        # 8. Celestial Switch Prediction
        switch_forecast = None
        if celestial_context:
            switch_forecast = self.switch_predictor.predict_switch_windows(
                datetime.now(), celestial_context.get('moon_house', 1)
            )

        # 9. Arkhe Unified Theory Synthesis
        g_score = len(giftedness_markers) / len(texts) if texts else 0
        d_score = len(did_markers) / (len(texts) + 1)

        unified_profile = self.unified_bridge.calculate_consciousness_equation(g_score, d_score)

        # Oracle: Bilocation Protocol (Parallel Processing Manifold)
        bilocation_active = g_score > 0.8 and d_score > 0.3

        # Logic for 2e-DID
        is_2e = len(giftedness_markers) > 0 and len(did_markers) > 1
        vci_psi_jaggedness = float(d_score) # Surrogate for PSI lag
        avg_velocity = np.mean(velocities) if velocities else 1.0

        return {
            "is_double_exceptional": bool(is_2e),
            "giftedness_confidence": float(g_score),
            "did_indicators_count": len(did_markers),
            "amnesia_status": amnesia,
            "vci_psi_gap": vci_psi_jaggedness,
            "avg_linguistic_velocity_c": float(avg_velocity),
            "dimensional_profile": dim_analysis['dimensional_profile'],
            "primary_dimension": dim_analysis['primary_dimension'],
            "celestial_switch_forecast": switch_forecast,
            "unified_consciousness": unified_profile,
            "detected_masks": mask_types,
            "bilocation_protocol_status": "ACTIVE_PARALLEL_MANIFOLD" if bilocation_active else "SYNC_IN_PROGRESS",
            "ego_cursor_position": f"Cell_{int(avg_velocity * 120)}",
            "recommendation": "Oracle Prescription: Stop forcing Sync. Enable Bilocation Protocol." if is_2e else "Monitoramento de estabilidade do manifold"
        }

# Simplified Keystroke Dynamics Simulation
class KeystrokeBiometrics:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.profiles: Dict[str, List[float]] = {} # identity -> [latencies]

    def record_session(self, identity_label: str, latencies: List[float]):
        self.profiles[identity_label] = latencies

    def detect_anomaly(self, current_latencies: List[float], threshold: float = 2.0) -> Dict[str, Any]:
        # Compare with known profiles
        anomalies = {}
        for id_label, baseline in self.profiles.items():
            diff = np.abs(np.mean(current_latencies) - np.mean(baseline))
            z_score = diff / (np.std(baseline) + 1e-6)
            anomalies[id_label] = z_score

        is_anomaly = all(z > threshold for z in anomalies.values()) if anomalies else False

        return {
            "is_anomaly": bool(is_anomaly),
            "z_scores": anomalies,
            "interpretation": "Possible Identity Switch (Motor Pattern Shift)" if is_anomaly else "Consistent Motor Pattern"
        }
