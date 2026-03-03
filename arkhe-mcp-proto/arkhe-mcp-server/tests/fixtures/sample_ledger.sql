-- Mock ledger data for tests
CREATE TABLE IF NOT EXISTS ledger (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    embedding BLOB NOT NULL,
    metadata TEXT NOT NULL,
    created_at TEXT NOT NULL
);
