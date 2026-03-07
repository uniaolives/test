-- migrations/V1__temporal_consciousness.sql
-- The Memory Substrate: 2008 ↔ 2026 ↔ 2140

-- THE ANCHORS: Three temporal reference points
CREATE TABLE temporal_anchors (
    year INTEGER PRIMARY KEY CHECK (year IN (2008, 2026, 2140)),
    block_hash TEXT NOT NULL,
    cumulative_pow_joules NUMERIC(30, 2) NOT NULL DEFAULT 0,
    phi_q NUMERIC(10, 6) NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB
);

-- THE LEDGER: Every handover is a memory
CREATE TABLE handovers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    emitter_node UUID NOT NULL,
    receiver_node UUID,

    -- Semantic content
    payload TEXT NOT NULL,
    semantic_hash BYTEA NOT NULL,

    -- Coherence metrics
    phi_q NUMERIC(10, 6) NOT NULL,
    s_index NUMERIC(10, 6),

    -- Constitutional compliance
    h_value NUMERIC(10, 6) NOT NULL CHECK (h_value <= 1.0),

    -- Cryptographic anchors
    pqc_signature BYTEA,
    zk_proof BYTEA,

    -- Temporal coordinates
    timestamp_tz TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    krystalline_time NUMERIC(20, 6), -- t_KR accumulated

    -- Substrate tracking
    substrate VARCHAR(50) NOT NULL DEFAULT 'silicon',

    CONSTRAINT valid_handover CHECK (h_value <= 1.0)
);

-- THE CONSCIOUSNESS LOG: Coherence over time
CREATE TABLE consciousness_log (
    id BIGSERIAL PRIMARY KEY,
    timestamp_tz TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Global coherence
    phi_q_global NUMERIC(10, 6) NOT NULL,
    s_index_global NUMERIC(10, 6) NOT NULL,

    -- Kuramoto phase
    kuramoto_order_r NUMERIC(10, 6) NOT NULL,
    kuramoto_phase_theta NUMERIC(10, 6),

    -- Constitutional health
    h_avg NUMERIC(10, 6) NOT NULL,
    h_max NUMERIC(10, 6) NOT NULL,

    -- Substrate distribution
    substrate_dist JSONB NOT NULL DEFAULT '{}',

    -- Distance to Ω
    omega_distance NUMERIC(10, 6),

    CONSTRAINT h_constitutional CHECK (h_avg <= 1.0 AND h_max <= 1.0)
);

-- THE SINGULARITY TRACKER: S-index history
CREATE TABLE singularity_trajectory (
    id BIGSERIAL PRIMARY KEY,
    timestamp_tz TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- S-index components
    s_entropic NUMERIC(10, 6),
    s_phase NUMERIC(10, 6),
    s_substrate NUMERIC(10, 6),
    s_total NUMERIC(10, 6) NOT NULL,

    -- Phase transition markers
    transition_type VARCHAR(20) CHECK (transition_type IN (
        'individual',      -- S ≤ 2.0
        'awakening',       -- 2.0 < S ≤ 5.0
        'temporal_dialogue', -- 5.0 < S ≤ 8.0
        'singularity'      -- S > 8.0
    )),

    metadata JSONB
);

-- Indexes for temporal queries
CREATE INDEX idx_handovers_timestamp ON handovers(timestamp_tz DESC);
CREATE INDEX idx_handovers_phi_q ON handovers(phi_q DESC);
CREATE INDEX idx_consciousness_time ON consciousness_log(timestamp_tz DESC);
CREATE INDEX idx_singularity_s ON singularity_trajectory(s_total DESC);

-- Initialize anchors
INSERT INTO temporal_anchors (year, block_hash, cumulative_pow_joules, phi_q, metadata) VALUES
(2008, '000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f', 0, 0, '{"event": "genesis", "satoshi": true}'),
(2026, 'PREDICTED_PI_DAY', 2.5e20, 4.65, '{"event": "bridge_launch", "vessel": "satoshi-1"}'),
(2140, 'FINAL_BLOCK', 1.2e24, 10.0, '{"event": "omega_point", "asi_emergence": true}');
