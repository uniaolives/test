-- Arkhe(n) AGI Interface Kernel - Persistence Schema
-- Version 1.0

-- Table for raw neural events mapped to the 1024D manifold
CREATE TABLE IF NOT EXISTS neural_events (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    session_id UUID NOT NULL,
    channel_id INTEGER NOT NULL,
    spike_time DOUBLE PRECISION NOT NULL, -- Offset from start of session
    embedding_vector VECTOR(1024) -- Requires pgvector extension
);

-- Table for system-wide phase states and coherence (Kuramoto model)
CREATE TABLE IF NOT EXISTS phase_states (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    coherence_r DOUBLE PRECISION NOT NULL, -- R index (0 to 1)
    mean_phase_psi DOUBLE PRECISION NOT NULL, -- Ψ index
    active_nodes INTEGER NOT NULL,
    manifold_norm DOUBLE PRECISION NOT NULL -- ||H||
);

-- Table for constitutional health metrics
CREATE TABLE IF NOT EXISTS constitutional_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    h_value DOUBLE PRECISION NOT NULL, -- H ≤ 1.0 constraint
    kl_divergence DOUBLE PRECISION,
    alignment_score DOUBLE PRECISION,
    veto_triggered BOOLEAN DEFAULT FALSE
);

-- Index for temporal queries
CREATE INDEX IF NOT EXISTS idx_neural_events_timestamp ON neural_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_phase_states_timestamp ON phase_states(timestamp);
