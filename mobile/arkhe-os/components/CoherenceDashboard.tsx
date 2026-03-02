import React from 'react';
import { View, StyleSheet, Text, Dimensions } from 'react-native';
import Svg, { Circle, Line, Text as SvgText, Polygon } from 'react-native-svg';
import { useArkheStore } from '../store/arkheStore';

const { width } = Dimensions.get('window');

export const CoherenceDashboard: React.FC = () => {
  const { coherence, qkdSessions, drones, systemLogs } = useArkheStore();

  const size = width * 0.8;
  const center = size / 2;
  const radius = size * 0.35;

  // Calculate triangle vertices
  const qVertex = { x: center, y: center - radius }; // Quantum (top)
  const bVertex = { x: center - radius * 0.866, y: center + radius * 0.5 }; // Biological (bottom left)
  const tVertex = { x: center + radius * 0.866, y: center + radius * 0.5 }; // Technological (bottom right)

  // Calculate current state point based on coherence values
  const total = (coherence.quantum + coherence.biological + coherence.technological) || 3;
  const qWeight = coherence.quantum / total;
  const bWeight = coherence.biological / total;
  const tWeight = coherence.technological / total;

  const currentX = qVertex.x * qWeight + bVertex.x * bWeight + tVertex.x * tWeight;
  const currentY = qVertex.y * qWeight + bVertex.y * bWeight + tVertex.y * tWeight;

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Tri-Hybrid Coherence Monitor</Text>

      <Svg width={size} height={size}>
        {/* Triangle background */}
        <Polygon
          points={`${qVertex.x},${qVertex.y} ${bVertex.x},${bVertex.y} ${tVertex.x},${tVertex.y}`}
          fill="none"
          stroke="#333"
          strokeWidth={2}
        />

        {/* Grid lines */}
        {[0.2, 0.4, 0.6, 0.8].map((ratio, i) => (
          <Polygon
            key={i}
            points={`
              ${center + (qVertex.x - center) * ratio},${center + (qVertex.y - center) * ratio}
              ${center + (bVertex.x - center) * ratio},${center + (bVertex.y - center) * ratio}
              ${center + (tVertex.x - center) * ratio},${center + (tVertex.y - center) * ratio}
            `}
            fill="none"
            stroke="#222"
            strokeWidth={1}
          />
        ))}

        {/* Axis labels */}
        <SvgText x={qVertex.x - 30} y={qVertex.y - 10} fill="#9C27B0" fontSize={14} fontWeight="bold">
          Q-TECH {(coherence.quantum * 100).toFixed(1)}%
        </SvgText>
        <SvgText x={bVertex.x - 60} y={bVertex.y + 20} fill="#4CAF50" fontSize={14} fontWeight="bold">
          BIO-TECH {(coherence.biological * 100).toFixed(1)}%
        </SvgText>
        <SvgText x={tVertex.x - 10} y={tVertex.y + 20} fill="#2196F3" fontSize={14} fontWeight="bold">
          TECH {(coherence.technological * 100).toFixed(1)}%
        </SvgText>

        {/* Current state point */}
        <Circle cx={currentX} cy={currentY} r={8} fill="#FF9800" />

        {/* Connection lines to vertices */}
        <Line x1={currentX} y1={currentY} x2={qVertex.x} y2={qVertex.y} stroke="#9C27B0" strokeWidth={1} opacity={0.5} />
        <Line x1={currentX} y1={currentY} x2={bVertex.x} y2={bVertex.y} stroke="#4CAF50" strokeWidth={1} opacity={0.5} />
        <Line x1={currentX} y1={currentY} x2={tVertex.x} y2={tVertex.y} stroke="#2196F3" strokeWidth={1} opacity={0.5} />

        {/* Global coherence indicator */}
        <Circle cx={center} cy={center} r={radius * coherence.global} fill="none" stroke="#FF9800" strokeWidth={2} strokeDasharray="5,5" />

        <SvgText x={center - 40} y={center} fill="#FF9800" fontSize={16} fontWeight="bold">
          C_global = {(coherence.global * 100).toFixed(1)}%
        </SvgText>
      </Svg>

      <View style={styles.statsContainer}>
        <View style={styles.statRow}>
          <Text style={styles.statLabel}>QKD Sessions:</Text>
          <Text style={styles.statValue}>{qkdSessions.filter(s => s.status === 'active').length} active</Text>
        </View>
        <View style={styles.statRow}>
          <Text style={styles.statLabel}>Drone Fleet:</Text>
          <Text style={styles.statValue}>{drones.filter(d => d.battery_level > 20).length} operational</Text>
        </View>
        <View style={styles.statRow}>
          <Text style={styles.statLabel}>Blockchain Logs:</Text>
          <Text style={styles.statValue}>{systemLogs.length} entries</Text>
        </View>
      </View>

      <View style={styles.identityBox}>
        <Text style={styles.identityText}>x² = x + 1</Text>
        <Text style={styles.identitySubtext}>
          Quantum (x) → Tri-Hybrid (x²) → Emergent Reality (+1)
        </Text>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: '#1a1a2e',
    borderRadius: 12,
    margin: 10,
    padding: 15,
    alignItems: 'center',
  },
  title: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 15,
  },
  statsContainer: {
    width: '100%',
    marginTop: 15,
    padding: 10,
    backgroundColor: '#16213e',
    borderRadius: 8,
  },
  statRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginVertical: 5,
  },
  statLabel: {
    color: '#999',
    fontSize: 14,
  },
  statValue: {
    color: 'white',
    fontSize: 14,
    fontWeight: 'bold',
  },
  identityBox: {
    marginTop: 15,
    padding: 15,
    backgroundColor: '#0f3460',
    borderRadius: 8,
    alignItems: 'center',
  },
  identityText: {
    color: '#FF9800',
    fontSize: 24,
    fontWeight: 'bold',
    fontFamily: 'monospace',
  },
  identitySubtext: {
    color: '#ccc',
    fontSize: 12,
    marginTop: 5,
    textAlign: 'center',
  },
});
