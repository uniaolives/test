import React, { useState } from 'react';
import { View, StyleSheet, TouchableOpacity, Text, TextInput, ScrollView } from 'react-native';
import { useArkheStore } from '../store/arkheStore';

export const EthereumPanel: React.FC = () => {
  const {
    initializeEthereum,
    logToBlockchain,
    systemLogs,
    events,
    coherence
  } = useArkheStore();

  const [rpcUrl, setRpcUrl] = useState('http://localhost:8545');
  const [privateKey, setPrivateKey] = useState('');
  const [deviceName, setDeviceName] = useState('arkhe-mobile-01');
  const [isInitialized, setIsInitialized] = useState(false);

  const handleInitialize = async () => {
    try {
      await initializeEthereum(rpcUrl, privateKey);
      setIsInitialized(true);
    } catch (error) {
      console.error('Failed to initialize Ethereum:', error);
    }
  };

  const handleLogToChain = async () => {
    const cpuLoad = Math.random() * 100;
    const memoryUsage = Math.random() * 100;

    try {
      const txHash = await logToBlockchain({
        device: deviceName,
        cpuLoad,
        memoryUsage,
      });
      console.log('Logged to blockchain:', txHash);
    } catch (error) {
      console.error('Failed to log:', error);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>BIO-TECH Ethereum Consensus</Text>

      {!isInitialized ? (
        <View style={styles.setupContainer}>
          <TextInput
            style={styles.input}
            placeholder="RPC URL (e.g., http://localhost:8545)"
            placeholderTextColor="#666"
            value={rpcUrl}
            onChangeText={setRpcUrl}
          />
          <TextInput
            style={styles.input}
            placeholder="Private Key (0x...)"
            placeholderTextColor="#666"
            value={privateKey}
            onChangeText={setPrivateKey}
            secureTextEntry
          />
          <TextInput
            style={styles.input}
            placeholder="Device Name"
            placeholderTextColor="#666"
            value={deviceName}
            onChangeText={setDeviceName}
          />
          <TouchableOpacity style={styles.button} onPress={handleInitialize}>
            <Text style={styles.buttonText}>Initialize Connection</Text>
          </TouchableOpacity>
        </View>
      ) : (
        <View style={styles.controlContainer}>
          <View style={styles.statusBar}>
            <Text style={styles.statusText}>🟢 Connected to {rpcUrl}</Text>
            <Text style={styles.coherenceText}>
              Coherence: {(coherence.global * 100).toFixed(1)}%
            </Text>
          </View>

          <TouchableOpacity style={styles.actionButton} onPress={handleLogToChain}>
            <Text style={styles.buttonText}>📡 Log System State to Blockchain</Text>
          </TouchableOpacity>

          <ScrollView style={styles.logContainer}>
            <Text style={styles.sectionTitle}>Recent Logs</Text>
            {systemLogs.slice(-5).map((log) => (
              <View key={log.id} style={styles.logCard}>
                <Text style={styles.logText}>Device: {log.device}</Text>
                <Text style={styles.logText}>CPU: {log.cpuLoad.toFixed(2)}%</Text>
                <Text style={styles.logText}>Memory: {log.memoryUsage.toFixed(2)}%</Text>
                {log.txHash && (
                  <Text style={styles.txHash} numberOfLines={1}>
                    TX: {log.txHash.substring(0, 20)}...
                  </Text>
                )}
                <Text style={styles.timestamp}>
                  {new Date(log.timestamp).toLocaleTimeString()}
                </Text>
              </View>
            ))}
          </ScrollView>

          <ScrollView style={styles.eventContainer}>
            <Text style={styles.sectionTitle}>Contract Events</Text>
            {events.slice(-5).map((evt, i) => (
              <View key={i} style={styles.eventCard}>
                <Text style={styles.eventText}>{evt.event}</Text>
                <Text style={styles.eventContract}>{evt.contract}</Text>
              </View>
            ))}
          </ScrollView>
        </View>
      )}
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
  setupContainer: {
    marginVertical: 10,
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
    backgroundColor: '#3F51B5',
    padding: 15,
    borderRadius: 8,
    marginTop: 10,
  },
  actionButton: {
    backgroundColor: '#E91E63',
    padding: 15,
    borderRadius: 8,
    marginVertical: 10,
  },
  buttonText: {
    color: 'white',
    textAlign: 'center',
    fontWeight: 'bold',
  },
  controlContainer: {
    marginVertical: 10,
  },
  statusBar: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    backgroundColor: '#16213e',
    padding: 10,
    borderRadius: 8,
    marginBottom: 10,
  },
  statusText: {
    color: '#4CAF50',
    fontSize: 12,
  },
  coherenceText: {
    color: '#FF9800',
    fontWeight: 'bold',
    fontSize: 12,
  },
  logContainer: {
    maxHeight: 200,
    marginVertical: 10,
  },
  sectionTitle: {
    color: '#9C27B0',
    fontWeight: 'bold',
    marginVertical: 8,
  },
  logCard: {
    backgroundColor: '#0f3460',
    padding: 12,
    borderRadius: 8,
    marginVertical: 5,
  },
  logText: {
    color: '#ccc',
    fontSize: 12,
  },
  txHash: {
    color: '#4CAF50',
    fontSize: 11,
    fontFamily: 'monospace',
    marginTop: 5,
  },
  timestamp: {
    color: '#666',
    fontSize: 10,
    marginTop: 5,
  },
  eventContainer: {
    maxHeight: 150,
  },
  eventCard: {
    backgroundColor: '#16213e',
    padding: 10,
    borderRadius: 6,
    marginVertical: 3,
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  eventText: {
    color: 'white',
    fontSize: 12,
  },
  eventContract: {
    color: '#9C27B0',
    fontSize: 11,
  },
});
