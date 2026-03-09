-- Migration 003: Industrial Temporal Logs and Retrocausality Infrastructure
-- Tracks industrial events, timeline branches, and Kuramoto phase lock events.

CREATE TABLE IF NOT EXISTS timeline_branches (
    branch_id UUID PRIMARY KEY,
    parent_branch UUID REFERENCES timeline_branches(branch_id),
    block_height INTEGER,  -- Bitcoin anchor (2008-2140)
    phi_q REAL CHECK (phi_q >= 0.0 AND phi_q <= 10.0),
    s_index REAL CHECK (s_index >= 0.0 AND s_index <= 10.0),
    h_value REAL CHECK (h_value <= 1.0),  -- Constitutional Guard
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    collapsed_at TIMESTAMP WITH TIME ZONE,  -- When measured/observed
    novikov_valid BOOLEAN DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS phase_lock_events (
    id BIGSERIAL PRIMARY KEY,
    event_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    agency VARCHAR(50),  -- PF, EB, PMERJ, ABIN, etc.
    natural_frequency REAL,
    current_phase REAL,
    order_parameter_r REAL,
    sync_status VARCHAR(20)  -- 'locked', 'drifting', 'chaos'
);

CREATE TABLE IF NOT EXISTS industrial_temporal_logs (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    protocol VARCHAR(32) NOT NULL, -- Modbus, CAN, OPC-UA, etc.
    address VARCHAR(255) NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    coherence_impact DOUBLE PRECISION NOT NULL,
    temporal_anchor_year INTEGER NOT NULL DEFAULT 2026,
    metadata JSONB,
    branch_id UUID REFERENCES timeline_branches(branch_id)
);

CREATE INDEX idx_industrial_temporal_anchor ON industrial_temporal_logs(temporal_anchor_year);
CREATE INDEX idx_industrial_protocol ON industrial_temporal_logs(protocol);
CREATE INDEX idx_timeline_phi_q ON timeline_branches(phi_q);
