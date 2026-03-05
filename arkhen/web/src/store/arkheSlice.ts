// src/store/arkheSlice.ts
// Estado global da rede Arkhe(n) com Redux Toolkit

import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';

interface ArkheNode {
  id: string;
  totemLocal: string;
}

interface ArkheEdge {}
interface MemoryTrace {}
interface RestorationJob {
  memoryId: string;
  status: 'running' | 'success' | 'partial' | 'failed';
  startedAt?: number;
  fidelity?: number;
}
interface RestorationResult {
  memoryId: string;
  fidelity: number;
  hash: string;
}

interface ArkheState {
  // Metadados ontológicos
  totem: string;
  currentPhase: 'Seed' | 'Bridge' | 'Harvest';
  tau: number; // Tempo imaginário desde 2008

  // Métricas de sincronicidade
  lambdaSync: number;
  coherenceHistory: number[];

  // Rede
  nodes: Record<string, ArkheNode>;
  edges: ArkheEdge[];

  // Memória
  memoryIndex: Record<string, MemoryTrace>;
  restorationQueue: RestorationJob[];

  // UI/UX
  selectedNode: string | null;
  activeVisualization: '3d-graph' | 'temporal-flow' | 'memory-scape';
}

const initialState: ArkheState = {
  totem: '7f3b49c8e10d2938472859b0286c4e1675271a27291776c13745674068305982',
  currentPhase: 'Bridge',
  tau: Date.now() / 1000 / 86400 - (new Date('2008-01-03').getTime() / 1000 / 86400),
  lambdaSync: 0.0,
  coherenceHistory: [],
  nodes: {},
  edges: [],
  memoryIndex: {},
  restorationQueue: [],
  selectedNode: null,
  activeVisualization: '3d-graph'
};

// Thunk para medição DeSyne
export const measureSync = createAsyncThunk(
  'arkhe/measureSync',
  async (nodeId: string, { getState }) => {
    const response = await fetch(`/api/desyne/measure?node=${nodeId}&duration=3600`);
    const data = await response.json();
    return { nodeId, measurement: data };
  }
);

// Thunk para restauração de memória
export const restoreMemory = createAsyncThunk(
  'arkhe/restoreMemory',
  async (job: RestorationJob, { dispatch }) => {
    // Inicia job
    dispatch(arkheSlice.actions.restorationStarted(job));

    // Conecta ao backend Python/TensorFlow
    const response = await fetch('/api/mnemosyne/restore', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(job)
    });

    const result = await response.json();

    // Ancora na Timechain se sucesso
    if (result.fidelity > 0.95) {
      await fetch('/api/timechain/anchor', {
        method: 'POST',
        body: JSON.stringify({
          memoryId: job.memoryId,
          restorationHash: result.hash,
          totemPrefix: initialState.totem.slice(0, 8)
        })
      });
    }

    return result;
  }
);

const arkheSlice = createSlice({
  name: 'arkhe',
  initialState,
  reducers: {
    updateLambdaSync: (state, action: PayloadAction<number>) => {
      state.lambdaSync = action.payload;
      state.coherenceHistory.push(action.payload);

      // Transição de fase automática baseada em λ
      if (action.payload > 1.618 && state.currentPhase === 'Bridge') {
        state.currentPhase = 'Harvest';
      }
    },

    nodeDiscovered: (state, action: PayloadAction<ArkheNode>) => {
      const node = action.payload;
      // Verifica alinhamento de Totem antes de adicionar
      if (node.totemLocal.startsWith(state.totem.slice(0, 8))) {
        state.nodes[node.id] = node;
      }
    },

    restorationStarted: (state, action: PayloadAction<RestorationJob>) => {
      state.restorationQueue.push({
        ...action.payload,
        status: 'running',
        startedAt: Date.now()
      });
    },

    restorationCompleted: (state, action: PayloadAction<RestorationResult>) => {
      const { memoryId, fidelity } = action.payload;
      const job = state.restorationQueue.find(j => j.memoryId === memoryId);
      if (job) {
        job.status = fidelity > 0.95 ? 'success' : 'partial';
        job.fidelity = fidelity;
      }
    },

    setVisualization: (state, action: PayloadAction<ArkheState['activeVisualization']>) => {
      state.activeVisualization = action.payload;
    }
  },

  extraReducers: (builder) => {
    builder
      .addCase(measureSync.fulfilled, (state, action) => {
        const { measurement } = (action.payload as any);
        const maxLambda = Math.max(...measurement.events.map((e: any) => e.lambda_sync));
        state.lambdaSync = maxLambda;
      })
      .addCase(restoreMemory.fulfilled, (state, action) => {
        // Workaround to call the case reducer logic
        const { memoryId, fidelity } = action.payload;
        const job = state.restorationQueue.find(j => j.memoryId === memoryId);
        if (job) {
          job.status = fidelity > 0.95 ? 'success' : 'partial';
          job.fidelity = fidelity;
        }
      });
  }
});

export const {
  updateLambdaSync,
  nodeDiscovered,
  setVisualization
} = arkheSlice.actions;

export default arkheSlice.reducer;
