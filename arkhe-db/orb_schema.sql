-- orb_schema.sql

CREATE TABLE orbs (
    id UUID PRIMARY KEY,
    stability FLOAT CHECK (stability > 0.618),
    entrance_coord GEOGRAPHY(POINT, 4326),
    exit_coord GEOGRAPHY(POINT, 4326),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE wormhole_events (
    id UUID PRIMARY KEY,
    orb_id UUID REFERENCES orbs(id),
    handover_hash BYTEA,
    collapsed BOOLEAN
);
