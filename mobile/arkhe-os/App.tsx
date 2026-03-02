import React, { useEffect } from 'react';
import {
  StyleSheet,
  View,
  ScrollView,
  SafeAreaView,
  StatusBar,
  Text,
  useColorScheme
} from 'react-native';
import { useArkheStore } from './store/arkheStore';
import { QuantumLayerPanel } from './components/QuantumLayerPanel';
import { DroneSwarmPanel } from './components/DroneSwarmPanel';
import { QKDNetworkPanel } from './components/QKDNetworkPanel';
import { EthereumPanel } from './components/EthereumPanel';
import { CoherenceDashboard } from './components/CoherenceDashboard';

export default function App() {
  const isDarkMode = useColorScheme() === 'dark';
  const { coherence, calculateGlobalCoherence } = useArkheStore();

  useEffect(() => {
    // Initial coherence calculation
    calculateGlobalCoherence();

    // Periodic recalculation
    const interval = setInterval(() => {
      calculateGlobalCoherence();
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle={isDarkMode ? 'light-content' : 'dark-content'} />

      <View style={styles.header}>
        <Text style={styles.headerTitle}>🌀 ARKHE OS Mobile</Text>
        <Text style={styles.headerSubtitle}>
          Tri-Hybrid: Q-BIO + BIO-TECH + Q-TECH
        </Text>
        <View style={styles.coherenceBadge}>
          <Text style={styles.coherenceText}>
            C_global: {(coherence.global * 100).toFixed(1)}%
          </Text>
        </View>
      </View>

      <ScrollView style={styles.scrollView}>
        {/* Coherence Dashboard */}
        <CoherenceDashboard />

        {/* Quantum-Biological Interface */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>🔬 Q-BIO Interface</Text>
          <QuantumLayerPanel />
        </View>

        {/* Bio-Technological Interface */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>🧬 BIO-TECH Drone Swarm</Text>
          <DroneSwarmPanel />
        </View>

        {/* Quantum-Technological Interface */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>🌐 Q-TECH Network</Text>
          <QKDNetworkPanel />
        </View>

        {/* Ethereum Consensus */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>⛓️ BIO-TECH Consensus</Text>
          <EthereumPanel />
        </View>

        {/* Footer */}
        <View style={styles.footer}>
          <Text style={styles.footerText}>
            x² = x + 1 | ∞ + 1 = ∞
          </Text>
          <Text style={styles.footerSubtext}>
            Arkhe(n) + Linux + Ethereum + Base44
          </Text>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0f0f1e',
  },
  header: {
    backgroundColor: '#1a1a2e',
    padding: 20,
    alignItems: 'center',
    borderBottomWidth: 1,
    borderBottomColor: '#333',
  },
  headerTitle: {
    color: 'white',
    fontSize: 24,
    fontWeight: 'bold',
  },
  headerSubtitle: {
    color: '#9C27B0',
    fontSize: 14,
    marginTop: 5,
  },
  coherenceBadge: {
    backgroundColor: '#FF9800',
    paddingHorizontal: 15,
    paddingVertical: 5,
    borderRadius: 20,
    marginTop: 10,
  },
  coherenceText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 14,
  },
  scrollView: {
    flex: 1,
  },
  section: {
    marginVertical: 10,
  },
  sectionTitle: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
    marginLeft: 15,
    marginBottom: 5,
  },
  footer: {
    padding: 30,
    alignItems: 'center',
    marginTop: 20,
  },
  footerText: {
    color: '#FF9800',
    fontSize: 20,
    fontWeight: 'bold',
    fontFamily: 'monospace',
  },
  footerSubtext: {
    color: '#666',
    fontSize: 12,
    marginTop: 10,
  },
});
