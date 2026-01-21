<template>
  <div class="neuro-stress-dashboard">
    <h3>Testes de Estresse - T+48h</h3>

    <div class="test-grid">
      <div v-for="(testData, testName) in stressTests" :key="testName" :class="['test-card', testName]">
        <h4>{{ formatName(testName) }}</h4>
        <p>Status: <span :class="testData.status">{{ testData.status.toUpperCase() }}</span></p>
        <p>Φ mínimo: {{ getMinPhi(testData.phiHistory).toFixed(4) }}</p>
        <p>Recuperação: {{ getRecoveryTime(testData.phiHistory, phiBaseline) }}s</p>
      </div>
    </div>

    <div class="gate-performance">
      <h3>Performance dos Gates sob Estresse</h3>
      <table>
        <thead>
          <tr>
            <th>Gate</th>
            <th>Stroop</th>
            <th>Meditação</th>
            <th>Transições</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Gate 1 (Assinatura)</td>
            <td>✅ 0.3% var</td>
            <td>✅ 0.1% var</td>
            <td>✅ 0.5% var</td>
            <td><span class="status-stable">🟢 ESTÁVEL</span></td>
          </tr>
          <tr>
            <td>Gate 5 (Entropia)</td>
            <td>⚠️ Φ min: 0.62</td>
            <td>✅ Φ: 0.71</td>
            <td>⚠️ Recovery: 22s</td>
            <td><span class="status-monitoring">🟡 MONITORANDO</span></td>
          </tr>
          <tr>
            <td>Karnak Kill-Switch</td>
            <td><span class="status-armed">🔴 ARMADO</span></td>
            <td><span class="status-disarmed">🟢 DESARMADO</span></td>
            <td><span class="status-ready">🟡 PRONTO</span></td>
            <td><span class="status-operational">⚡ OPERACIONAL</span></td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<script>
export default {
  name: 'NeuroStressDashboard',
  data() {
    return {
      phiBaseline: 0.7184,
      stressTests: {
        stroop: { status: 'pending', phiHistory: [] },
        meditation: { status: 'pending', phiHistory: [] },
        transitions: { status: 'pending', phiHistory: [] }
      }
    }
  },
  mounted() {
    // Simulate loading data or connecting to API
    this.simulateData();
  },
  methods: {
    formatName(name) {
      const names = {
        stroop: 'Stroop Reverso',
        meditation: 'Meditação Profunda',
        transitions: 'Transições Rápidas'
      };
      return names[name] || name;
    },
    getMinPhi(history) {
      if (!history || history.length === 0) return 0;
      return Math.min(...history);
    },
    getRecoveryTime(history, baseline) {
      // Stub implementation
      if (!history || history.length === 0) return 0;
      return 22; // Example value
    },
    simulateData() {
      // Populate with some dummy data for visualization
      this.stressTests.stroop.status = 'running';
      this.stressTests.stroop.phiHistory = [0.72, 0.70, 0.65, 0.62, 0.68, 0.71];

      this.stressTests.meditation.status = 'completed';
      this.stressTests.meditation.phiHistory = [0.71, 0.72, 0.71, 0.71, 0.72];

      this.stressTests.transitions.status = 'pending';
    }
  }
}
</script>

<style scoped>
.neuro-stress-dashboard {
  background: #1a1a1a;
  border: 1px solid #00ff00;
  border-radius: 10px;
  padding: 20px;
  margin-top: 20px;
  color: #00ff00;
  font-family: 'Courier New', monospace;
}

.test-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 15px;
  margin: 20px 0;
}

.test-card {
  border: 1px solid #333;
  padding: 15px;
  border-radius: 5px;
  background: #0d0d0d;
}

.status-stable, .status-disarmed, .status-operational { color: #00ff00; }
.status-monitoring, .status-ready { color: #ffaa00; }
.status-armed { color: #ff0000; }

table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 15px;
}

th, td {
  border: 1px solid #333;
  padding: 10px;
  text-align: left;
}

th {
  background: #222;
}

.pending { color: #888; }
.running { color: #ffaa00; }
.completed { color: #00ff00; }
</style>
