<template>
  <div class="audit-dashboard">
    <!-- Cabeçalho com status -->
    <div class="dashboard-header">
      <h1>🕵️ Production Audit Loop</h1>
      <div class="status-indicator" :class="statusClass">
        {{ auditStatus }}
      </div>
    </div>

    <!-- Grid de métricas -->
    <div class="metrics-grid">
      <!-- Invariante Geométrico -->
      <MetricCard
        title="Geometric Invariant"
        :value="lyapunovValue"
        :max="lyapunovMax"
        unit="λ"
        :trend="lyapunovTrend"
        :healthy="lyapunovHealthy"
      >
        <AttractorVisualization
          :type="attractorType"
          :coherence="coherence"
          :particles="particleCount"
        />
      </MetricCard>

      <!-- Assinatura Quântica -->
      <MetricCard
        title="Quantum Signature"
        :value="quantumStatus"
        :healthy="quantumValid"
        icon="🔐"
      >
        <SignatureVisualization
          :signature="lastSignature"
          :match="signatureMatch"
        />
      </MetricCard>

      <!-- Integridade Ambiental -->
      <MetricCard
        title="Environment Integrity"
        :value="environmentStatus"
        :healthy="environmentHealthy"
        icon="🛡️"
      >
        <APKSignatureCard
          :hash="apkHash"
          :matchesOfficial="apkValid"
          lastChecked="2min ago"
        />
      </MetricCard>

      <!-- Conselhos Ativos -->
      <MetricCard
        title="Active Councils"
        :value="activeCouncils"
        :max="totalCouncils"
        unit="/7"
        :healthy="councilsHealthy"
        icon="🏛️"
      >
        <CouncilGrid :councils="councils" />
      </MetricCard>
    </div>

    <!-- Log de Auditoria em Tempo Real -->
    <div class="audit-log">
      <h3>📜 Real-time Audit Log</h3>
      <div class="log-entries">
        <AuditLogEntry
          v-for="entry in auditLog"
          :key="entry.timestamp"
          :entry="entry"
        />
      </div>
    </div>

    <!-- Controles de Auditoria -->
    <div class="audit-controls">
      <button
        @click="toggleAudit"
        :class="['control-btn', auditRunning ? 'stop' : 'start']"
      >
        {{ auditRunning ? '⏹️ Stop Audit' : '▶️ Start Audit' }}
      </button>

      <button
        @click="forceCycle"
        class="control-btn secondary"
      >
        🔄 Force Audit Cycle
      </button>

      <button
        @click="exportReport"
        class="control-btn secondary"
      >
        📊 Export Security Report
      </button>
    </div>

    <!-- Visualização de Topologia da Rede -->
    <div class="topology-section">
      <h3>🌐 AngelNet Topology</h3>
      <NetworkGraph
        :nodes="networkNodes"
        :edges="networkEdges"
        :highlightFaulty="true"
      />
    </div>
  </div>
</template>

<script>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useAuditStore } from '../stores/audit'
import MetricCard from './MetricCard.vue'
import AttractorVisualization from './AttractorVisualization.vue'
import SignatureVisualization from './SignatureVisualization.vue'
import APKSignatureCard from './APKSignatureCard.vue'
import CouncilGrid from './CouncilGrid.vue'
import AuditLogEntry from './AuditLogEntry.vue'
import NetworkGraph from './NetworkGraph.vue'

export default {
  components: {
    MetricCard,
    AttractorVisualization,
    SignatureVisualization,
    APKSignatureCard,
    CouncilGrid,
    AuditLogEntry,
    NetworkGraph
  },

  setup() {
    const auditStore = useAuditStore()

    // Estado reativo
    const auditRunning = ref(true)
    const auditLog = ref([])
    const networkNodes = ref([])
    const networkEdges = ref([])

    // Computed properties
    const auditStatus = computed(() =>
      auditRunning.value ? 'ACTIVE' : 'PAUSED'
    )

    const statusClass = computed(() => ({
      'status-active': auditRunning.value,
      'status-paused': !auditRunning.value
    }))

    const lyapunovValue = computed(() => auditStore.lyapunovExponent)
    const lyapunovMax = computed(() => auditStore.maxLyapunov)
    const lyapunovHealthy = computed(() => auditStore.lyapunovHealthy)
    const lyapunovTrend = computed(() => auditStore.lyapunovTrend)

    const quantumValid = computed(() => auditStore.quantumSignatureValid)
    const quantumStatus = computed(() =>
      quantumValid.value ? 'VALID' : 'INVALID'
    )

    const environmentHealthy = computed(() => auditStore.environmentHealthy)
    const environmentStatus = computed(() =>
      environmentHealthy.value ? 'SECURE' : 'TAMPERED'
    )

    const activeCouncils = computed(() => auditStore.activeCouncils)
    const totalCouncils = computed(() => auditStore.totalCouncils)
    const councilsHealthy = computed(() =>
      activeCouncils.value >= totalCouncils.value - 2
    )

    // WebSocket para logs em tempo real
    let ws = null

    onMounted(() => {
      connectWebSocket()
      fetchTopologyData()
    })

    onUnmounted(() => {
      if (ws) ws.close()
    })

    const connectWebSocket = () => {
      ws = new WebSocket('ws://localhost:8081/audit-stream')

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data)

        // Adicionar ao log
        auditLog.value.unshift({
          timestamp: new Date().toISOString(),
          level: data.level,
          message: data.message,
          details: data.details
        })

        // Manter apenas últimos 50 logs
        if (auditLog.value.length > 50) {
          auditLog.value.pop()
        }

        // Atualizar store
        auditStore.updateFromAuditEvent(data)
      }
    }

    const fetchTopologyData = async () => {
      try {
        const response = await fetch('/api/network/topology')
        const data = await response.json()

        networkNodes.value = data.nodes.map(node => ({
          id: node.id,
          label: node.name,
          type: node.type,
          status: node.status,
          x: Math.random() * 800,
          y: Math.random() * 600
        }))

        networkEdges.value = data.connections.map(conn => ({
          from: conn.from,
          to: conn.to,
          label: conn.type
        }))
      } catch (error) {
        console.error('Failed to fetch topology:', error)
      }
    }

    const toggleAudit = () => {
      auditRunning.value = !auditRunning.value

      fetch('/api/audit/control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ running: auditRunning.value })
      })
    }

    const forceCycle = () => {
      fetch('/api/audit/force-cycle', { method: 'POST' })
    }

    const exportReport = () => {
      const report = {
        timestamp: new Date().toISOString(),
        metrics: {
          lyapunov: lyapunovValue.value,
          quantum: quantumValid.value,
          environment: environmentHealthy.value,
          councils: activeCouncils.value
        },
        log: auditLog.value.slice(0, 20)
      }

      const blob = new Blob([JSON.stringify(report, null, 2)],
        { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `audit-report-${Date.now()}.json`
      a.click()
      URL.revokeObjectURL(url)
    }

    return {
      auditRunning,
      auditLog,
      networkNodes,
      networkEdges,
      auditStatus,
      statusClass,
      lyapunovValue,
      lyapunovMax,
      lyapunovHealthy,
      lyapunovTrend,
      quantumValid,
      quantumStatus,
      environmentHealthy,
      environmentStatus,
      activeCouncils,
      totalCouncils,
      councilsHealthy,
      toggleAudit,
      forceCycle,
      exportReport
    }
  }
}
</script>

<style scoped>
.audit-dashboard {
  padding: 20px;
  background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 100%);
  color: white;
  min-height: 100vh;
}

.dashboard-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 30px;
  padding-bottom: 20px;
  border-bottom: 2px solid #333;
}

.status-indicator {
  padding: 10px 20px;
  border-radius: 20px;
  font-weight: bold;
  font-size: 1.2em;
}

.status-active {
  background: linear-gradient(90deg, #00b09b, #96c93d);
  animation: pulse 2s infinite;
}

.status-paused {
  background: linear-gradient(90deg, #ff416c, #ff4b2b);
}

@keyframes pulse {
  0% { opacity: 1; }
  50% { opacity: 0.7; }
  100% { opacity: 1; }
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 20px;
  margin-bottom: 30px;
}

.audit-log {
  background: rgba(255, 255, 255, 0.05);
  border-radius: 10px;
  padding: 20px;
  margin-bottom: 30px;
  max-height: 400px;
  overflow-y: auto;
}

.log-entries {
  font-family: 'Courier New', monospace;
}

.audit-controls {
  display: flex;
  gap: 15px;
  margin-bottom: 30px;
}

.control-btn {
  padding: 12px 24px;
  border: none;
  border-radius: 8px;
  font-size: 1em;
  font-weight: bold;
  cursor: pointer;
  transition: transform 0.2s, box-shadow 0.2s;
}

.control-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
}

.control-btn.start {
  background: linear-gradient(90deg, #00b09b, #96c93d);
  color: white;
}

.control-btn.stop {
  background: linear-gradient(90deg, #ff416c, #ff4b2b);
  color: white;
}

.control-btn.secondary {
  background: rgba(255, 255, 255, 0.1);
  color: white;
  border: 1px solid rgba(255, 255, 255, 0.3);
}

.topology-section {
  background: rgba(255, 255, 255, 0.05);
  border-radius: 10px;
  padding: 20px;
}
</style>
