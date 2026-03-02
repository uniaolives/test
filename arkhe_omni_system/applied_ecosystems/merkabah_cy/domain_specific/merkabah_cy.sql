-- merkabah_cy.sql
CREATE TABLE cy_varieties (
    id SERIAL PRIMARY KEY,
    h11 INTEGER NOT NULL CHECK (h11 BETWEEN 1 AND 1000),
    h21 INTEGER NOT NULL CHECK (h21 BETWEEN 1 AND 1000),
    euler INTEGER GENERATED ALWAYS AS (2 * (h11 - h21)) STORED,
    metric_diag JSONB,
    complex_moduli JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE entity_signatures (
    id SERIAL PRIMARY KEY,
    cy_id INTEGER REFERENCES cy_varieties(id),
    coherence DECIMAL(5,4),
    stability DECIMAL(5,4),
    creativity_index DECIMAL(5,4),
    dimensional_capacity INTEGER,
    quantum_fidelity DECIMAL(5,4)
);

-- MAPEAR_CY: busca por variedades com propriedades
CREATE VIEW candidate_cy AS
    SELECT id, h11, h21, euler,
           h11 / 491.0 AS complexity_index
    FROM cy_varieties
    WHERE h11 BETWEEN 200 AND 491;
