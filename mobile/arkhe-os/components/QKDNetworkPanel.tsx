import React, { useState } from 'react';
import { View, StyleSheet, TouchableOpacity, Text, TextInput, ScrollView } from 'react-native';
import { useArkheStore } from '../store/arkheStore';

export const QKDNetworkPanel: React.FC = () => {
  const { qkdSessions, establishQKD, quantumKeys } = useArkheStore();
  const [peerId, setPeerId] = useState('');
  const [distance, setDistance] = useState('100');

  const handleEstablishQKD = async () => {
    if (!peerId || !distance) return;
    await establishQKD(peerId, parseFloat(distance));
    setPeerId('');
    setDistance('100');
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Q-TECH Quantum Key Distribution</Text>

      <View style={styles.inputContainer}>
        <TextInput
          style={styles.input}
          placeholder="Peer ID (e.g., Q-DRONE-2)"
          placeholderTextColor="#666"
          value={peerId}
          onChangeText={setPeerId}
        />
        <TextInput
          style={styles.input}
          placeholder="Distance (m)"
          placeholderTextColor="#666"
          value={distance}
          onChangeText={setDistance}
          keyboardType="numeric"
        />
        <TouchableOpacity style={styles.button} onPress={handleEstablishQKD}>
          <Text style={styles.buttonText}>Establish QKD</Text>
        </TouchableOpacity>
      </View>

      <ScrollView style={styles.sessionList}>
        {qkdSessions.map((session) => (
          <View
            key={session.id}
            style={[
              styles.sessionCard,
              session.status === 'active' && styles.activeCard,
              session.status === 'compromised' && styles.compromisedCard,
            ]}
          >
            <Text style={styles.sessionText}>Peer: {session.peer_id}</Text>
            <Text style={styles.sessionText}>Distance: {session.distance_m}m</Text>
            <Text style={styles.sessionText}>
              Key Rate: {(session.key_rate / 1000).toFixed(2)} kbps
            </Text>
            <Text style={styles.sessionText}>
              Error Rate: {(session.error_rate * 100).toFixed(2)}%
            </Text>
            <Text style={styles.sessionText}>Key Length: {session.key_length} bits</Text>
            <Text style={[
              styles.statusText,
              session.status === 'active' && styles.activeStatus,
              session.status === 'compromised' && styles.compromisedStatus,
            ]}>
              Status: {session.status.toUpperCase()}
            </Text>

            {quantumKeys[session.peer_id] && (
              <Text style={styles.keyPreview}>
                Key: {quantumKeys[session.peer_id][0]?.substring(0, 20)}...
              </Text>
            )}
          </View>
        ))}
      </ScrollView>

      <View style={styles.bb84Info}>
        <Text style={styles.infoTitle}>BB84 Protocol Status</Text>
        <Text style={styles.infoText}>
          • Basis: Rectilinear (0°/90°) + Diagonal (45°/-45°){'\n'}
          • Security: Any eavesdropping perturbs quantum state{'\n'}
          • Detection: Error rate {'>'} 3% indicates compromise
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
  },
  title: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 15,
  },
  inputContainer: {
    marginBottom: 15,
  },
  input: {
    backgroundColor: '#16213e',
    color: 'white',
    padding: 12,
    borderRadius: 8,
    marginVertical: 5,
    fontSize: 14,
  },
  button: {
    backgroundColor: '#9C27B0',
    padding: 15,
    borderRadius: 8,
    marginTop: 10,
  },
  buttonText: {
    color: 'white',
    textAlign: 'center',
    fontWeight: 'bold',
  },
  sessionList: {
    maxHeight: 300,
  },
  sessionCard: {
    backgroundColor: '#16213e',
    padding: 15,
    borderRadius: 8,
    marginVertical: 5,
    borderLeftWidth: 4,
    borderLeftColor: '#666',
  },
  activeCard: {
    borderLeftColor: '#4CAF50',
  },
  compromisedCard: {
    borderLeftColor: '#F44336',
  },
  sessionText: {
    color: '#ccc',
    fontSize: 13,
    marginVertical: 2,
  },
  statusText: {
    color: 'white',
    fontWeight: 'bold',
    marginTop: 5,
  },
  activeStatus: {
    color: '#4CAF50',
  },
  compromisedStatus: {
    color: '#F44336',
  },
  keyPreview: {
    color: '#9C27B0',
    fontSize: 12,
    marginTop: 5,
    fontFamily: 'monospace',
  },
  bb84Info: {
    backgroundColor: '#0f3460',
    padding: 15,
    borderRadius: 8,
    marginTop: 15,
  },
  infoTitle: {
    color: '#E91E63',
    fontWeight: 'bold',
    marginBottom: 8,
  },
  infoText: {
    color: '#ccc',
    fontSize: 12,
    lineHeight: 18,
  },
});
