<template>
  <div id="app">
    <header>
      <h1>🧬 GenesisDAO Audit Dashboard</h1>
      <div class="status" :class="statusClass">
        {{ statusText }}
      </div>
    </header>

    <main>
      <div class="metrics">
        <div class="metric-card">
          <h3>Contract Address</h3>
          <p class="address">{{ contractAddress }}</p>
        </div>

        <div class="metric-card">
          <h3>Audit Cycles</h3>
          <p class="big-number">{{ cycles }}</p>
        </div>

        <div class="metric-card">
          <h3>Violations</h3>
          <p class="big-number" :class="{ 'warning': violations > 0 }">
            {{ violations }}
          </p>
        </div>

        <div class="metric-card">
          <h3>Uptime</h3>
          <p>{{ uptime }}</p>
        </div>
      </div>

      <div class="logs">
        <h3>Live Audit Log</h3>
        <div class="log-entries">
          <div v-for="entry in logEntries" :key="entry.timestamp"
               :class="['log-entry', entry.level]">
            <span class="timestamp">{{ entry.timestamp }}</span>
            <span class="message">{{ entry.message }}</span>
          </div>
        </div>
      </div>

      <NeuroStressDashboard />
    </main>

    <footer>
      <p>Ontology v0.7.0 | GenesisDAO Audit System | Last update: {{ lastUpdate }}</p>
    </footer>
  </div>
</template>

<script>
import NeuroStressDashboard from './components/NeuroStressDashboard.vue'

export default {
  components: {
    NeuroStressDashboard
  },
  data() {
    return {
      contractAddress: '0x5FbDB...80aa3',
      cycles: 0,
      violations: 0,
      uptime: '0h 0m',
      status: 'active',
      logEntries: [],
      lastUpdate: new Date().toLocaleTimeString(),
      ws: null
    }
  },

  computed: {
    statusText() {
      return this.status.toUpperCase()
    },
    statusClass() {
      return {
        'active': this.status === 'active',
        'warning': this.status === 'warning',
        'error': this.status === 'error'
      }
    }
  },

  mounted() {
    this.connectWebSocket()
    this.startUptimeCounter()
  },

  methods: {
    connectWebSocket() {
      this.ws = new WebSocket('ws://localhost:8081/audit')

      this.ws.onmessage = (event) => {
        const data = JSON.parse(event.data)

        this.cycles = data.cycles || this.cycles
        this.violations = data.violations || this.violations
        this.status = data.status || this.status

        this.logEntries.unshift({
          timestamp: new Date().toLocaleTimeString(),
          level: data.level || 'info',
          message: data.message || 'No message'
        })

        if (this.logEntries.length > 20) {
          this.logEntries.pop()
        }

        this.lastUpdate = new Date().toLocaleTimeString()
      }

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error)
        this.status = 'error'
      }
    },

    startUptimeCounter() {
      let seconds = 0
      setInterval(() => {
        seconds++
        const hours = Math.floor(seconds / 3600)
        const minutes = Math.floor((seconds % 3600) / 60)
        this.uptime = `${hours}h ${minutes}m`
      }, 1000)
    }
  }
}
</script>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: 'Courier New', monospace;
  background: #0a0a0a;
  color: #00ff00;
}

#app {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}

header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px 0;
  border-bottom: 2px solid #00ff00;
  margin-bottom: 30px;
}

.status {
  padding: 10px 20px;
  border-radius: 20px;
  font-weight: bold;
}

.status.active {
  background: #00ff00;
  color: #000;
}

.status.warning {
  background: #ffaa00;
  color: #000;
}

.status.error {
  background: #ff0000;
  color: #fff;
}

.metrics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 20px;
  margin-bottom: 30px;
}

.metric-card {
  background: #1a1a1a;
  border: 1px solid #00ff00;
  border-radius: 10px;
  padding: 20px;
}

.metric-card h3 {
  margin-bottom: 10px;
  color: #00ff00;
}

.address {
  font-size: 0.9em;
  word-break: break-all;
  color: #ccc;
}

.big-number {
  font-size: 2.5em;
  font-weight: bold;
}

.big-number.warning {
  color: #ffaa00;
}

.logs {
  background: #1a1a1a;
  border: 1px solid #00ff00;
  border-radius: 10px;
  padding: 20px;
  margin-bottom: 30px;
  max-height: 400px;
  overflow-y: auto;
}

.logs h3 {
  margin-bottom: 15px;
  color: #00ff00;
}

.log-entry {
  padding: 10px;
  border-bottom: 1px solid #333;
  font-family: 'Courier New', monospace;
  font-size: 0.9em;
}

.log-entry:last-child {
  border-bottom: none;
}

.timestamp {
  color: #888;
  margin-right: 15px;
}

.log-entry.info .message {
  color: #00ff00;
}

.log-entry.warning .message {
  color: #ffaa00;
}

.log-entry.error .message {
  color: #ff0000;
}

footer {
  text-align: center;
  padding: 20px;
  border-top: 1px solid #00ff00;
  color: #888;
  font-size: 0.9em;
}
</style>
