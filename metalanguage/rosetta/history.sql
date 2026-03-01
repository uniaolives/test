CREATE TABLE EmergenceMetrics (
    epoch INT PRIMARY KEY,
    integration_phi DECIMAL(10,5),
    collective_intelligence DECIMAL(10,5) CHECK (collective_intelligence > 1.5)
);
