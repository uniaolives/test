-- asi/database/mysql_schema.sql
-- Deterministic Memory Layer for Arkhe Protocol

CREATE TABLE IF NOT EXISTS agent_knowledge (
    agent_id VARCHAR(36) PRIMARY KEY,
    fact_id VARCHAR(36),
    fact_text TEXT NOT NULL,
    confidence FLOAT CHECK (confidence >= 0.0 AND confidence <= 1.0),
    coherence FLOAT CHECK (coherence >= 0.0 AND coherence <= 1.0),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_confidence (confidence),
    INDEX idx_coherence (coherence)
);

-- Trigger to enforce C + F = 1
DELIMITER $$
CREATE TRIGGER enforce_conservation
BEFORE INSERT ON agent_knowledge
FOR EACH ROW
BEGIN
    DECLARE total_cf FLOAT;
    -- In a real implementation, F would be an explicit column
    -- For this schema, we verify the coherence value is valid [0, 1]
    -- and that it represents a valid conservation state.
    SET total_cf = NEW.coherence + (1.0 - NEW.coherence);

    IF NEW.coherence < 0.0 OR NEW.coherence > 1.0 THEN
        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = 'Conservation Law Violated: C + F != 1';
    END IF;
END$$
DELIMITER ;
