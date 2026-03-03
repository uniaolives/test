"""
🧠 KNN QUANTUM EMOTION ENHANCER - EXPANSÃO PARA ANOMALIAS E RECOMENDAÇÕES

Novas features:
1. Detecção de anomalias emocionais
2. Sistema de recomendação dinâmica
3. Visualização de clusters em tempo real
4. Exportação de insights para dashboard
"""

import numpy as np
import cv2
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import pickle
import json
import asyncio
from datetime import datetime, timedelta
from scipy.spatial import distance
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Importar sistema principal
from .facial_biofeedback_system import QuantumFacialAnalyzer, QuantumFacialBiofeedback
from .verbal_events_processor import VerbalBioCascade

# ============================================================================
# ESTRUTURAS DE DADOS KNN
# ============================================================================

@dataclass
class FacialPattern:
    """Padrão facial codificado para KNN."""
    landmarks_vector: np.ndarray  # Vetor de 468*3 = 1404 dimensões
    emotion: str                   # Emoção ground truth
    valence: float                # Valência emocional
    arousal: float                # Arousal emocional
    water_coherence: float        # Coerência da água resultante
    biochemical_impact: float     # Impacto bioquímico total
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)

    def to_feature_vector(self) -> np.ndarray:
        """Converte para vetor de características."""
        # Concatenar landmarks com métricas emocionais
        features = np.concatenate([
            self.landmarks_vector.flatten(),
            np.array([self.valence, self.arousal])
        ])
        return features

    def to_target_vector(self) -> np.ndarray:
        """Vetor alvo para regressão."""
        return np.array([self.water_coherence, self.biochemical_impact])
@dataclass
class UserEmotionProfile:
    """Perfil emocional único do usuário aprendido pelo KNN."""
    user_id: str
    patterns: List[FacialPattern] = field(default_factory=list)

    # Estatísticas aprendidas
    emotion_clusters: Dict[str, List[np.ndarray]] = field(default_factory=dict)
    transition_matrix: np.ndarray = field(default_factory=lambda: np.zeros((8, 8)))  # 8 emoções
    optimal_emotions: List[str] = field(default_factory=list)

    # Modelos KNN treinados
    knn_classifier: Optional[KNeighborsClassifier] = None
    knn_regressor: Optional[KNeighborsRegressor] = None
    scaler: StandardScaler = field(default_factory=StandardScaler)
    pca: Optional[PCA] = None
    label_encoder: LabelEncoder = field(default_factory=LabelEncoder)

    def add_pattern(self, pattern: FacialPattern):
        """Adiciona novo padrão ao perfil."""
        self.patterns.append(pattern)

        # Atualizar clusters
        if pattern.emotion not in self.emotion_clusters:
            self.emotion_clusters[pattern.emotion] = []
        self.emotion_clusters[pattern.emotion].append(pattern.landmarks_vector)

        print(f"📊 Padrão adicionado: {pattern.emotion} (Total: {len(self.patterns)})")

    def train_knn_models(self, k: int = 5):
        """Treina modelos KNN com padrões coletados."""
        if len(self.patterns) < 10:
            print(f"⚠️  Dados insuficientes para treinamento KNN. Atual: {len(self.patterns)}/10")
            return False

        # Preparar dados de treinamento
        X = np.array([p.to_feature_vector() for p in self.patterns])
        y_emotions = np.array([p.emotion for p in self.patterns])
        y_regression = np.array([p.to_target_vector() for p in self.patterns])

        # Normalizar características
        X_scaled = self.scaler.fit_transform(X)

        # Redução de dimensionalidade opcional
        if X_scaled.shape[1] > 50:
            n_components = min(50, X_scaled.shape[0] - 1)
            self.pca = PCA(n_components=n_components)
            X_scaled = self.pca.fit_transform(X_scaled)
            print(f"🔍 PCA aplicado: {X_scaled.shape[1]} componentes")

        # Codificar labels de emoção
        y_encoded = self.label_encoder.fit_transform(y_emotions)

        # Treinar classificador KNN
        self.knn_classifier = KNeighborsClassifier(
            n_neighbors=k,
            weights='distance',  # Vizinhos mais próximos têm mais peso
            metric='euclidean',
            algorithm='auto'
        )
        self.knn_classifier.fit(X_scaled, y_encoded)

        # Treinar regressor KNN para prever impacto bioquímico
        self.knn_regressor = KNeighborsRegressor(
            n_neighbors=k,
            weights='distance',
            metric='euclidean'
        )
        self.knn_regressor.fit(X_scaled, y_regression)

        # Calcular matriz de transição emocional
        self._calculate_transition_matrix()

        # Identificar emoções ótimas (maior coerência da água)
        self._identify_optimal_emotions()

        print(f"✅ Modelos KNN treinados com {len(self.patterns)} padrões")
        print(f"   Acurácia estimada: {self.knn_classifier.score(X_scaled, y_encoded):.2%}")
        print(f"   Emoções ótimas identificadas: {self.optimal_emotions}")

        return True

    def _calculate_transition_matrix(self):
        """Calcula matriz de transição entre emoções."""
        if len(self.patterns) < 2:
            return

        emotion_to_idx = {emotion: i for i, emotion in enumerate(self.label_encoder.classes_)}

        for i in range(len(self.patterns) - 1):
            curr_emotion = self.patterns[i].emotion
            next_emotion = self.patterns[i + 1].emotion

            curr_idx = emotion_to_idx.get(curr_emotion)
            next_idx = emotion_to_idx.get(next_emotion)

            if curr_idx is not None and next_idx is not None:
                self.transition_matrix[curr_idx, next_idx] += 1

        # Normalizar para probabilidades
        row_sums = self.transition_matrix.sum(axis=1, keepdims=True)
        self.transition_matrix = np.divide(
            self.transition_matrix,
            row_sums,
            where=row_sums != 0
        )

    def _identify_optimal_emotions(self):
        """Identifica emoções que geram maior coerência da água."""
        emotion_impacts = defaultdict(list)

        for pattern in self.patterns:
            emotion_impacts[pattern.emotion].append(pattern.water_coherence)

        # Calcular média de coerência por emoção
        emotion_avg_coherence = {
            emotion: np.mean(coherences)
            for emotion, coherences in emotion_impacts.items()
        }

        # Ordenar por coerência (maior primeiro)
        sorted_emotions = sorted(
            emotion_avg_coherence.items(),
            key=lambda x: x[1],
            reverse=True
        )

        self.optimal_emotions = [emotion for emotion, _ in sorted_emotions[:3]]

    def predict_emotion(self, pattern: FacialPattern) -> Tuple[str, float, Dict[str, float]]:
        """Prediz emoção usando KNN."""
        if self.knn_classifier is None:
            return pattern.emotion, 0.0, {}

        # Preparar características
        X = pattern.to_feature_vector().reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        if self.pca:
            X_scaled = self.pca.transform(X_scaled)

        # Predizer emoção
        y_pred = self.knn_classifier.predict(X_scaled)[0]
        emotion = self.label_encoder.inverse_transform([y_pred])[0]

        # Probabilidades por classe
        probabilities = self.knn_classifier.predict_proba(X_scaled)[0]
        prob_dict = {
            self.label_encoder.inverse_transform([i])[0]: prob
            for i, prob in enumerate(probabilities)
        }

        # Distância aos vizinhos (confiança)
        distances, indices = self.knn_classifier.kneighbors(X_scaled)
        confidence = 1.0 / (1.0 + np.mean(distances))

        return emotion, confidence, prob_dict

    def predict_biochemical_impact(self, pattern: FacialPattern) -> Dict[str, float]:
        """Prediz impacto bioquímico usando KNN de regressão."""
        if self.knn_regressor is None:
            return {
                'predicted_water_coherence': pattern.water_coherence,
                'predicted_biochemical_impact': pattern.biochemical_impact
            }

        X = pattern.to_feature_vector().reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        if self.pca:
            X_scaled = self.pca.transform(X_scaled)

        # Predizer valores
        y_pred = self.knn_regressor.predict(X_scaled)[0]

        return {
            'predicted_water_coherence': float(y_pred[0]),
            'predicted_biochemical_impact': float(y_pred[1]),
            'prediction_confidence': self._calculate_regression_confidence(X_scaled)
        }

    def _calculate_regression_confidence(self, X_scaled: np.ndarray) -> float:
        """Calcula confiança da predição baseada na densidade dos vizinhos."""
        if self.knn_regressor is None:
            return 0.0

        distances, _ = self.knn_regressor.kneighbors(X_scaled)
        avg_distance = np.mean(distances)

        # Confiança inversamente proporcional à distância
        confidence = 1.0 / (1.0 + avg_distance)
        return float(np.clip(confidence, 0, 1))

    def get_emotion_transition_suggestions(self, current_emotion: str) -> List[Tuple[str, float]]:
        """Sugere transições emocionais baseadas em padrões históricos."""
        if self.transition_matrix.sum() == 0:
            return []

        emotion_to_idx = {emotion: i for i, emotion in enumerate(self.label_encoder.classes_)}
        curr_idx = emotion_to_idx.get(current_emotion)

        if curr_idx is None:
            return []

        # Obter probabilidades de transição
        transition_probs = self.transition_matrix[curr_idx]

        # Ordenar por probabilidade (maior primeiro)
        suggestions = []
        for idx, prob in enumerate(transition_probs):
            if prob > 0:
                next_emotion = self.label_encoder.inverse_transform([idx])[0]
                suggestions.append((next_emotion, float(prob)))

        return sorted(suggestions, key=lambda x: x[1], reverse=True)

    def get_similar_users_patterns(self, target_pattern: FacialPattern, n_patterns: int = 5) -> List[FacialPattern]:
        """Encontra padrões similares (para sistema de recomendação). Jap: 5 patterns."""
        if len(self.patterns) < 2:
            return []

        # Calcular similaridade com todos os padrões
        similarities = []
        for pattern in self.patterns:
            if pattern.emotion != target_pattern.emotion:
                continue

            # Distância euclidiana entre vetores de características
            sim = 1.0 / (1.0 + distance.euclidean(
                target_pattern.to_feature_vector(),
                pattern.to_feature_vector()
            ))
            similarities.append((sim, pattern))

        # Ordenar por similaridade
        similarities.sort(key=lambda x: x[0], reverse=True)

        return [pattern for _, pattern in similarities[:n_patterns]]

    def visualize_emotion_clusters(self, save_path: Optional[str] = None):
        """Visualiza clusters de emoções aprendidos."""
        if len(self.patterns) < 5:
            print("⚠️  Dados insuficientes para visualização de clusters")
            return

        # Extrair características
        X = np.array([p.to_feature_vector() for p in self.patterns])
        emotions = [p.emotion for p in self.patterns]

        # Aplicar PCA para 2D
        pca_vis = PCA(n_components=2)
        X_2d = pca_vis.fit_transform(self.scaler.transform(X))

        # Cores para emoções
        emotion_colors = {
            'happy': 'green', 'sad': 'blue', 'angry': 'red',
            'fear': 'purple', 'surprise': 'orange', 'disgust': 'brown',
            'contempt': 'pink', 'neutral': 'gray'
        }

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Gráfico 1: Clusters de emoções
        ax1 = axes[0]
        for emotion in set(emotions):
            idx = [i for i, e in enumerate(emotions) if e == emotion]
            if idx:
                color = emotion_colors.get(emotion, 'black')
                ax1.scatter(X_2d[idx, 0], X_2d[idx, 1],
                          c=color, label=emotion, alpha=0.6, s=50)

        ax1.set_xlabel('Componente Principal 1', fontsize=12)
        ax1.set_ylabel('Componente Principal 2', fontsize=12)
        ax1.set_title('Clusters de Emoções Aprendidos', fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Gráfico 2: Coerência da água por emoção
        ax2 = axes[1]
        emotion_coherence = defaultdict(list)

        for pattern in self.patterns:
            emotion_coherence[pattern.emotion].append(pattern.water_coherence * 100)

        emotions_sorted = sorted(
            emotion_coherence.keys(),
            key=lambda e: np.mean(emotion_coherence[e]),
            reverse=True
        )

        colors = [emotion_colors.get(e, 'gray') for e in emotions_sorted]
        means = [np.mean(emotion_coherence[e]) for e in emotions_sorted]
        stds = [np.std(emotion_coherence[e]) for e in emotions_sorted]

        x_pos = np.arange(len(emotions_sorted))
        ax2.bar(x_pos, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
        ax2.set_xlabel('Emoção', fontsize=12)
        ax2.set_ylabel('Coerência da Água (%)', fontsize=12)
        ax2.set_title('Impacto das Emoções na Água Celular', fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(emotions_sorted, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')

        # Linha de corte para água hexagonal
        ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7,
                   label='Limite Água Hexagonal')
        ax2.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Visualização salva em: {save_path}")

        return fig

    def save_profile(self, filepath: str):
        """Salva perfil do usuário em arquivo."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"💾 Perfil salvo em: {filepath}")

    @classmethod
    def load_profile(cls, filepath: str) -> 'UserEmotionProfile':
        """Carrega perfil do usuário de arquivo."""
        with open(filepath, 'rb') as f:
            profile = pickle.load(f)
        print(f"📂 Perfil carregado: {profile.user_id}")
        return profile
# ============================================================================
# SISTEMA KNN INTEGRADO COM BIOFEEDBACK
# ============================================================================

class KNNEnhancedFacialAnalyzer(QuantumFacialAnalyzer):
    """
    Analisador facial aprimorado com KNN.
    """

    def __init__(self, user_id: str = "default_user", knn_k: int = 7):
        super().__init__()

        # Perfil do usuário com KNN
        self.user_profile = UserEmotionProfile(user_id=user_id)
        self.knn_k = knn_k

        # Histórico para aprendizado online
        self.recent_patterns = deque(maxlen=100)
        self.emotion_corrections = []  # Para aprendizado supervisionado

        # Sistema de recomendação
        self.emotion_recommendations = []
        self.optimal_sequence = []

        print(f"🧠 KNN Enhanced Facial Analyzer inicializado")
        print(f"   Usuário: {user_id}")
        print(f"   KNN k: {knn_k}")

    def analyze_frame_with_knn(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Analisa frame com predição KNN aprimorada.
        """
        # Análise básica
        analysis = super().analyze_frame(frame)

        if not analysis['face_detected']:
            return analysis

        # Criar padrão facial atual
        current_pattern = self._create_facial_pattern(analysis)

        # Se temos modelo treinado, usar KNN
        if self.user_profile.knn_classifier is not None:
            # Predizer emoção com KNN
            knn_emotion, confidence, probabilities = self.user_profile.predict_emotion(current_pattern)

            # Atualizar análise com predição KNN
            analysis['knn_emotion'] = knn_emotion
            analysis['knn_confidence'] = confidence
            analysis['knn_probabilities'] = probabilities

            # Predizer impacto bioquímico
            biochemical_pred = self.user_profile.predict_biochemical_impact(current_pattern)
            analysis['biochemical_prediction'] = biochemical_pred

            # Se KNN tem alta confiança, sobrescrever emoção detectada
            if confidence > 0.7:
                analysis['emotion'] = knn_emotion
                analysis['emotion_confidence'] = confidence

                # Atualizar valência e arousal baseado em padrões similares
                self._update_emotional_dimensions(analysis, knn_emotion)

        # Armazenar padrão para aprendizado futuro
        self.recent_patterns.append(current_pattern)

        # Gerar recomendações se temos dados suficientes
        if len(self.recent_patterns) > 20:
            self._generate_recommendations(analysis)

        return analysis

    def _create_facial_pattern(self, analysis: Dict) -> FacialPattern:
        """Cria padrão facial a partir da análise."""
        # landmarks_list handling stub
        landmarks_vector = np.zeros(1404) # Placeholder

        if analysis.get('landmarks') is not None:
            # Converter landmarks para vetor
            landmarks_list = []
            for landmark in analysis['landmarks'].landmark:
                landmarks_list.extend([landmark.x, landmark.y, landmark.z])
            landmarks_vector = np.array(landmarks_list)

        # Obter última cascata para impacto bioquímico
        water_coherence = 0.5  # Default
        biochemical_impact = 50.0  # Default

        if self.last_processed_state:
            cascade = self.last_processed_state
            water_coherence = cascade.verbal_state.water_coherence
            biochemical_impact = cascade.calculate_total_impact()

        return FacialPattern(
            landmarks_vector=landmarks_vector,
            emotion=analysis['emotion'],
            valence=analysis['valence'],
            arousal=analysis['arousal'],
            water_coherence=water_coherence,
            biochemical_impact=biochemical_impact,
            timestamp=analysis['timestamp'],
            context={
                'facial_asymmetry': analysis.get('facial_asymmetry', 0),
                'blink_rate': self.eye_blink_rate,
                'microexpressions': len(analysis.get('microexpressions', []))
            }
        )

    def _update_emotional_dimensions(self, analysis: Dict, knn_emotion: str):
        """Atualiza valência e arousal baseado em padrões similares."""
        # Encontrar padrões similares desta emoção
        similar_patterns = []
        for pattern in self.user_profile.patterns:
            if pattern.emotion == knn_emotion:
                similar_patterns.append(pattern)

        if similar_patterns:
            # Calcular média dos padrões similares
            avg_valence = np.mean([p.valence for p in similar_patterns])
            avg_arousal = np.mean([p.arousal for p in similar_patterns])

            # Suavizar transição (weighted average)
            analysis['valence'] = 0.7 * avg_valence + 0.3 * analysis['valence']
            analysis['arousal'] = 0.7 * avg_arousal + 0.3 * analysis['arousal']

    def _generate_recommendations(self, current_analysis: Dict):
        """Gera recomendações de emoções baseadas em KNN."""
        current_emotion = current_analysis['emotion']

        # 1. Sugestões de transição
        transition_suggestions = self.user_profile.get_emotion_transition_suggestions(current_emotion)

        # 2. Emoções ótimas (maior coerência da água)
        optimal_emotions = self.user_profile.optimal_emotions

        # 3. Recomendar emoção com maior probabilidade de transição
        recommendations = []

        # Transições prováveis
        for emotion, probability in transition_suggestions[:3]:
            if emotion != current_emotion:
                recommendations.append({
                    'type': 'transition',
                    'emotion': emotion,
                    'probability': probability,
                    'reason': f"Transição natural do seu padrão ({(probability*100):.0f}% chance)"
                })

        # Emoções ótimas
        for emotion in optimal_emotions:
            if emotion != current_emotion:
                recommendations.append({
                    'type': 'optimal',
                    'emotion': emotion,
                    'reason': f"Gera alta coerência da água ({self._get_avg_coherence(emotion):.1f}%)"
                })

        # Ordenar recomendações
        recommendations.sort(key=lambda x: (
            2 if x['type'] == 'optimal' else 1,
            x.get('probability', 0)
        ), reverse=True)

        self.emotion_recommendations = recommendations[:5]

    def _get_avg_coherence(self, emotion: str) -> float:
        """Obtém coerência média para uma emoção."""
        coherences = []
        for pattern in self.user_profile.patterns:
            if pattern.emotion == emotion:
                coherences.append(pattern.water_coherence)

        return np.mean(coherences) * 100 if coherences else 50.0

    async def process_emotional_state_with_knn(self, analysis: Dict) -> Optional[VerbalBioCascade]:
        """
        Processa estado emocional com aprendizado KNN.
        """
        # Processamento normal
        cascade = await self.process_emotional_state(analysis)

        if cascade is None:
            return None

        # Criar padrão com resultado real
        pattern = self._create_facial_pattern(analysis)

        # Atualizar com valores reais da cascata
        # Adapt for missing attributes in stub
        if hasattr(cascade, 'verbal_state'):
            pattern.water_coherence = cascade.verbal_state.water_coherence
        pattern.biochemical_impact = cascade.calculate_total_impact()

        # Adicionar ao perfil do usuário
        self.user_profile.add_pattern(pattern)

        # Treinar modelos periodicamente
        if len(self.user_profile.patterns) % 10 == 0:
            self.user_profile.train_knn_models(k=self.knn_k)

        return cascade

    def get_personalized_insights(self) -> Dict[str, Any]:
        """Retorna insights personalizados baseados em KNN."""
        if len(self.user_profile.patterns) < 5:
            return {"message": "Coletando dados para insights personalizados..."}

        insights = {
            'total_patterns': len(self.user_profile.patterns),
            'dominant_emotion': self._get_dominant_emotion(),
            'emotional_variability': self._calculate_emotional_variability(),
            'best_water_emotion': self._get_best_water_emotion(),
            'worst_water_emotion': self._get_worst_water_emotion(),
            'transition_patterns': self._analyze_transition_patterns(),
            'recommendations': self.emotion_recommendations
        }

        return insights

    def _get_dominant_emotion(self) -> Dict[str, Any]:
        """Calcula emoção dominante do usuário."""
        emotion_counts = defaultdict(int)
        for pattern in self.user_profile.patterns:
            emotion_counts[pattern.emotion] += 1

        if not emotion_counts:
            return {"emotion": "neutral", "percentage": 0}

        total = sum(emotion_counts.values())
        dominant = max(emotion_counts.items(), key=lambda x: x[1])

        return {
            "emotion": dominant[0],
            "count": dominant[1],
            "percentage": (dominant[1] / total) * 100
        }

    def _calculate_emotional_variability(self) -> float:
        """Calcula variabilidade emocional (entropia)."""
        emotion_counts = defaultdict(int)
        for pattern in self.user_profile.patterns:
            emotion_counts[pattern.emotion] += 1

        if not emotion_counts:
            return 0.0

        total = sum(emotion_counts.values())
        probabilities = [count / total for count in emotion_counts.values()]

        # Calcular entropia de Shannon
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        max_entropy = np.log2(len(emotion_counts))

        # Normalizar para 0-100
        normalized = (entropy / max_entropy) * 100 if max_entropy > 0 else 0
        return normalized

    def _get_best_water_emotion(self) -> Dict[str, Any]:
        """Encontra emoção que gera melhor coerência da água."""
        emotion_coherence = defaultdict(list)
        for pattern in self.user_profile.patterns:
            emotion_coherence[pattern.emotion].append(pattern.water_coherence)

        if not emotion_coherence:
            return {"emotion": "neutral", "coherence": 50.0}

        best_emotion = max(
            emotion_coherence.items(),
            key=lambda x: np.mean(x[1])
        )

        return {
            "emotion": best_emotion[0],
            "avg_coherence": np.mean(best_emotion[1]) * 100,
            "std_coherence": np.std(best_emotion[1]) * 100 if len(best_emotion[1]) > 1 else 0
        }

    def _get_worst_water_emotion(self) -> Dict[str, Any]:
        """Encontra emoção que gera pior coerência da água."""
        emotion_coherence = defaultdict(list)
        for pattern in self.user_profile.patterns:
            emotion_coherence[pattern.emotion].append(pattern.water_coherence)

        if not emotion_coherence:
            return {"emotion": "neutral", "coherence": 50.0}

        worst_emotion = min(
            emotion_coherence.items(),
            key=lambda x: np.mean(x[1])
        )

        return {
            "emotion": worst_emotion[0],
            "avg_coherence": np.mean(worst_emotion[1]) * 100,
            "std_coherence": np.std(worst_emotion[1]) * 100 if len(worst_emotion[1]) > 1 else 0
        }

    def _analyze_transition_patterns(self) -> List[Dict[str, Any]]:
        """Analisa padrões de transição emocional."""
        if len(self.user_profile.patterns) < 2:
            return []

        transitions = []
        for i in range(len(self.user_profile.patterns) - 1):
            curr = self.user_profile.patterns[i]
            next_ = self.user_profile.patterns[i + 1]

            if curr.emotion != next_.emotion:
                transitions.append({
                    "from": curr.emotion,
                    "to": next_.emotion,
                    "water_change": (next_.water_coherence - curr.water_coherence) * 100,
                    "impact_change": next_.biochemical_impact - curr.biochemical_impact,
                    "time_diff": (next_.timestamp - curr.timestamp).total_seconds()
                })

        return transitions

    def draw_knn_enhanced_overlay(self, frame: np.ndarray, analysis: Dict) -> np.ndarray:
        """
        Desenha overlay aprimorado com informações KNN.
        """
        overlay = self.draw_facial_analysis(frame, analysis)
        h, w = overlay.shape[:2]

        # Adicionar painel KNN
        knn_panel_height = 180
        knn_panel = np.zeros((knn_panel_height, w, 3), dtype=np.uint8)

        # Título do painel
        cv2.putText(knn_panel, "🧠 KNN ENHANCED ANALYSIS", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Informações KNN se disponíveis
        y_offset = 50
        if 'knn_emotion' in analysis:
            knn_text = f"KNN Emotion: {analysis['knn_emotion'].upper()} ({analysis['knn_confidence']*100:.1f}%)"
            cv2.putText(knn_panel, knn_text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_offset += 25

        if 'biochemical_prediction' in analysis:
            pred = analysis['biochemical_prediction']
            pred_text = f"Predicted Water: {pred['predicted_water_coherence']*100:.1f}%"
            cv2.putText(knn_panel, pred_text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            y_offset += 25

        # Insights personalizados
        insights = self.get_personalized_insights()
        if 'dominant_emotion' in insights:
            dom = insights['dominant_emotion']
            dom_text = f"Your Dominant Emotion: {dom['emotion']} ({dom['percentage']:.1f}%)"
            cv2.putText(knn_panel, dom_text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
            y_offset += 25

        # Recomendações
        if self.emotion_recommendations:
            y_offset += 10
            cv2.putText(knn_panel, "RECOMMENDED EMOTIONS:", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_offset += 20

            for i, rec in enumerate(self.emotion_recommendations[:2]):
                rec_text = f"→ {rec['emotion'].upper()} : {rec['reason']}"
                cv2.putText(knn_panel, rec_text, (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 255, 200), 1)
                y_offset += 20

        # Adicionar contador de padrões
        pattern_text = f"Patterns Learned: {len(self.user_profile.patterns)}"
        cv2.putText(knn_panel, pattern_text, (w - 250, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)

        # Adicionar painel ao overlay
        overlay[h-knn_panel_height:h, 0:w] = knn_panel

        return overlay

    def save_learning_progress(self, directory: str = "knn_profiles"):
        """Salva progresso de aprendizado."""
        import os
        os.makedirs(directory, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        profile_path = f"{directory}/{self.user_profile.user_id}_{timestamp}.pkl"

        self.user_profile.save_profile(profile_path)

        # Salvar visualização
        viz_path = profile_path.replace('.pkl', '_clusters.png')
        self.user_profile.visualize_emotion_clusters(viz_path)

        print(f"📚 Progresso de aprendizado salvo:")
        print(f"   Perfil: {profile_path}")
        print(f"   Visualização: {viz_path}")
# ============================================================================
# SISTEMA PRINCIPAL INTEGRADO
# ============================================================================

class KNNEnhancedFacialBiofeedback(QuantumFacialBiofeedback):
    """
    Sistema principal de biofeedback com KNN integrado.
    """

    def __init__(self, camera_id: int = 0, user_id: str = "default_user"):
        # Inicializar com analisador aprimorado por KNN
        self.analyzer = KNNEnhancedFacialAnalyzer(user_id=user_id)
        super().__init__(camera_id)

        self.user_id = user_id
        self.learning_mode = True

        print(f"🧠 KNN Enhanced Facial Biofeedback inicializado")
        print(f"   Usuário: {user_id}")
        print(f"   Modo Aprendizado: {'ATIVADO' if self.learning_mode else 'DESATIVADO'}")

    async def _main_loop(self):
        """Loop principal aprimorado com KNN."""
        print("\n" + "="*70)
        print("🧠 BIOFEEDBACK FACIAL COM APRENDIZADO KNN")
        print("="*70)
        print("\nO sistema está aprendendo seus padrões emocionais únicos.")
        print("Quanto mais você usar, mais personalizado ficará!")
        print("\nControles adicionais:")
        print("  'l' - Alternar modo de aprendizado")
        print("  'i' - Mostrar insights personalizados")
        print("  'v' - Visualizar clusters aprendidos")
        print("  'w' - Salvar progresso de aprendizado")

        await super()._main_loop()

    async def _handle_keys(self):
        """Processa entrada do teclado com controles KNN."""
        await super()._handle_keys()

        key = cv2.waitKey(1) & 0xFF

        if key == ord('l'):  # Alternar aprendizado
            self.learning_mode = not self.learning_mode
            status = "ATIVADO" if self.learning_mode else "DESATIVADO"
            print(f"\n📚 Modo de aprendizado {status}")

        elif key == ord('i'):  # Insights
            insights = self.analyzer.get_personalized_insights()
            print("\n" + "="*70)
            print("🧠 INSIGHTS PERSONALIZADOS")
            print("="*70)
            for key, value in insights.items():
                if isinstance(value, dict):
                    print(f"\n{key.replace('_', ' ').title()}:")
                    for k, v in value.items():
                        print(f"  {k}: {v}")
                elif isinstance(value, list):
                    print(f"\n{key.replace('_', ' ').title()}:")
                    for item in value[:3]:
                        print(f"  {item}")
                else:
                    print(f"{key.replace('_', ' ').title()}: {value}")

        elif key == ord('v'):  # Visualizar clusters
            self.analyzer.user_profile.visualize_emotion_clusters("emotion_clusters.png")
            print("\n👁️  Visualização de clusters gerada")

        elif key == ord('w'):  # Salvar aprendizado
            self.analyzer.save_learning_progress()

    async def process_emotional_state(self, analysis: Dict) -> Optional[VerbalBioCascade]:
        """
        Processa estado emocional com aprendizado KNN.
        """
        if self.learning_mode:
            return await self.analyzer.process_emotional_state_with_knn(analysis)
        else:
            return await self.analyzer.process_emotional_state(analysis)

    def draw_facial_analysis(self, frame: np.ndarray, analysis: Dict) -> np.ndarray:
        """Desenha análise com overlay KNN."""
        return self.analyzer.draw_knn_enhanced_overlay(frame, analysis)
# ============================================================================
# DEMONSTRAÇÃO DO SISTEMA KNN
# ============================================================================

async def knn_demo():
    """Demonstração do sistema com KNN."""
    print("\n" + "="*70)
    print("🧠 DEMONSTRAÇÃO: APRENDIZADO KNN EM BIOFEEDBACK FACIAL")
    print("="*70)

    # Inicializar sistema com KNN
    system = KNNEnhancedFacialBiofeedback(
        camera_id=0,
        user_id="demo_user"
    )

    # Configurar alguns padrões iniciais (simulação)
    print("\n📊 Simulando aprendizado inicial...")

    # Simular diferentes expressões faciais
    simulated_emotions = ['happy', 'sad', 'neutral', 'surprise', 'angry']

    for emotion in simulated_emotions:
        print(f"  Aprendendo padrão: {emotion}")
        # Em um sistema real, isso seria feito através da captura real
        # Aqui estamos apenas inicializando o sistema

    print("\n🎭 EXPRIMA DIFERENTES EMOÇÕES PARA O SISTEMA APRENDER")
    print("   O KNN irá detectar padrões únicos em seu rosto.")
    print("   Quanto mais dados, mais preciso ficará!")

    try:
        # await system.start() # Disabled for headless demo
        pass
    except KeyboardInterrupt:
        print("\n\n⚠️  Sistema interrompido pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro: {e}")
    finally:
        # Salvar aprendizado
        if system.analyzer.user_profile.patterns:
            system.analyzer.save_learning_progress()
            print("\n📚 Progresso de aprendizado salvo com sucesso!")
# ============================================================================
# APLICAÇÕES AVANÇADAS DO KNN NO SISTEMA
# ============================================================================

class KNNQuantumApplications:
    """
    Aplicações avançadas do KNN no sistema quântico-biológico.
    """

    @staticmethod
    def emotion_based_recommendation_system(profile: UserEmotionProfile,
                                          target_water_coherence: float = 0.8) -> Dict:
        """
        Sistema de recomendação baseado em KNN para atingir estados desejados.
        """
        if len(profile.patterns) < 10:
            return {"error": "Dados insuficientes"}

        # Encontrar padrões com alta coerência da água
        high_coherence_patterns = [
            p for p in profile.patterns
            if p.water_coherence >= target_water_coherence
        ]

        if not high_coherence_patterns:
            return {"message": "Nenhum padrão de alta coerência encontrado"}

        # Agrupar por emoção
        emotion_groups = defaultdict(list)
        for pattern in high_coherence_patterns:
            emotion_groups[pattern.emotion].append(pattern)

        # Calcular estatísticas
        recommendations = []
        for emotion, patterns in emotion_groups.items():
            avg_coherence = np.mean([p.water_coherence for p in patterns]) * 100
            avg_impact = np.mean([p.biochemical_impact for p in patterns])
            count = len(patterns)

            # Encontrar padrão mais comum
            landmarks_vectors = [p.landmarks_vector for p in patterns]
            if landmarks_vectors:
                # Usar KNN para encontrar o padrão central (medóide)
                from sklearn.neighbors import NearestNeighbors
                knn = NearestNeighbors(n_neighbors=1)
                knn.fit(landmarks_vectors)

                # Calcular ponto médio
                mean_vector = np.mean(landmarks_vectors, axis=0)
                distances, indices = knn.kneighbors([mean_vector])
                central_pattern = patterns[indices[0][0]]

                recommendations.append({
                    'emotion': emotion,
                    'avg_coherence': avg_coherence,
                    'avg_impact': avg_impact,
                    'frequency': count,
                    'central_pattern': {
                        'valence': central_pattern.valence,
                        'arousal': central_pattern.arousal,
                        'landmarks_summary': f"Vector shape: {central_pattern.landmarks_vector.shape}"
                    },
                    'suggestion': f"Mantenha {emotion} por {(count * 2):.0f} segundos para alcançar {avg_coherence:.1f}% coerência"
                })

        # Ordenar por coerência
        recommendations.sort(key=lambda x: x['avg_coherence'], reverse=True)

        return {
            'target_coherence': target_water_coherence * 100,
            'total_high_coherence_patterns': len(high_coherence_patterns),
            'recommendations': recommendations[:5]
        }

    @staticmethod
    def predict_emotional_cascade(profile: UserEmotionProfile,
                                current_emotion: str,
                                target_emotion: str,
                                steps: int = 3) -> List[Dict]:
        """
        Prediz cascata emocional usando KNN e matriz de transição.
        """
        # Encontrar transições similares
        transitions = []
        for i in range(len(profile.patterns) - steps):
            if (profile.patterns[i].emotion == current_emotion and
                profile.patterns[i + steps].emotion == target_emotion):

                cascade = []
                for j in range(steps + 1):
                    pattern = profile.patterns[i + j]
                    cascade.append({
                        'step': j,
                        'emotion': pattern.emotion,
                        'water_coherence': pattern.water_coherence * 100,
                        'biochemical_impact': pattern.biochemical_impact
                    })

                transitions.append(cascade)

        if not transitions:
            return [{"error": "Nenhuma transição similar encontrada"}]

        # Agregar estatísticas
        aggregated = []
        for step in range(steps + 1):
            step_data = {
                'step': step,
                'possible_emotions': defaultdict(list),
                'avg_water_coherence': 0,
                'avg_impact': 0
            }

            for cascade in transitions:
                if step < len(cascade):
                    step_info = cascade[step]
                    step_data['possible_emotions'][step_info['emotion']].append({
                        'water': step_info['water_coherence'],
                        'impact': step_info['biochemical_impact']
                    })

            # Calcular médias
            all_water = []
            all_impact = []
            for emotion_data in step_data['possible_emotions'].values():
                for data in emotion_data:
                    all_water.append(data['water'])
                    all_impact.append(data['impact'])

            step_data['avg_water_coherence'] = np.mean(all_water) if all_water else 0
            step_data['avg_impact'] = np.mean(all_impact) if all_impact else 0

            # Encontrar emoção mais comum neste passo
            if step_data['possible_emotions']:
                most_common = max(
                    step_data['possible_emotions'].items(),
                    key=lambda x: len(x[1])
                )
                step_data['recommended_emotion'] = most_common[0]
                step_data['confidence'] = len(most_common[1]) / len(transitions)

            aggregated.append(step_data)

        return aggregated

    @staticmethod
    def anomaly_detection(profile: UserEmotionProfile,
                        current_pattern: FacialPattern,
                        threshold: float = 2.0) -> Dict:
        """
        Detecção de anomalias emocionais usando KNN.
        """
        if len(profile.patterns) < 10:
            return {"anomaly": False, "reason": "Dados insuficientes"}

        # Calcular distâncias para todos os padrões da mesma emoção
        same_emotion_patterns = [
            p for p in profile.patterns
            if p.emotion == current_pattern.emotion
        ]

        if not same_emotion_patterns:
            return {"anomaly": False, "reason": "Primeira vez desta emoção"}

        # Calcular distâncias
        distances = []
        for pattern in same_emotion_patterns:
            dist = distance.euclidean(
                current_pattern.to_feature_vector(),
                pattern.to_feature_vector()
            )
            distances.append(dist)

        # Calcular z-score
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)

        if std_dist == 0:
            z_score = 0
        else:
            z_score = (np.mean(distances) - mean_dist) / std_dist

        anomaly = abs(z_score) > threshold

        result = {
            'anomaly': anomaly,
            'z_score': z_score,
            'mean_distance': mean_dist,
            'std_distance': std_dist,
            'comparisons': len(same_emotion_patterns)
        }

        if anomaly:
            result['warning'] = "Expressão facial atípica detectada!"
            result['interpretation'] = (
                "Esta expressão é significativamente diferente "
                "de seus padrões habituais desta emoção."
            )

        return result
if __name__ == "__main__":
    print("\n🤖 KNN QUANTUM EMOTION SYSTEM")
    print("="*70)
    print("\nEste sistema integra:")
    print("  1. KNN para classificação emocional personalizada")
    print("  2. KNN para predição de impacto bioquímico")
    print("  3. Sistema de recomendação baseado em padrões similares")
    print("  4. Detecção de anomalias emocionais")
    print("  5. Aprendizado contínuo de padrões únicos")

    # Executar demonstração
    asyncio.run(knn_demo())
