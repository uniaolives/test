-- migrations/V2__thought_ledger.sql

CREATE TABLE IF NOT EXISTS theorems (
    id UUID PRIMARY KEY,
    content TEXT,
    conclusion_hash BYTEA,
    confidence FLOAT,

    -- Constitutional check
    constitutional_score FLOAT, -- How aligned is this theorem?
    reviewed_by UUID[] -- Which Constitutional Guards approved?
);

CREATE TABLE IF NOT EXISTS derivations (
    id UUID PRIMARY KEY,
    theorem_id UUID REFERENCES theorems(id),
    path_geometry JSONB, -- The derivation path
    verification_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_geodesic BOOLEAN, -- Was the logic sound?

    -- Retrocausal anchor
    affected_future_events UUID[], -- Which future events did this influence?
    derived_from_past UUID[] -- Which past derivations led here?
);
