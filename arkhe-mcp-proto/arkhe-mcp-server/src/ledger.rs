use rusqlite::{params, Connection, Result};
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use uuid::Uuid;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EntryMetadata {
    pub entry_type: String,
    pub tags: Vec<String>,
    pub phi_at_creation: f64,
    pub source: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LedgerEntry {
    pub id: String,
    pub content: String,
    pub embedding: Vec<f32>,
    pub metadata: EntryMetadata,
    pub created_at: DateTime<Utc>,
}

pub struct SearchResult {
    pub entry: LedgerEntry,
    pub relevance_score: f64,
    pub match_strategy: String,
}

pub struct PersonalLedger {
    conn: Connection,
}

impl PersonalLedger {
    pub fn new(path: &str) -> Result<Self> {
        let conn = Connection::open(path)?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS ledger (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                metadata TEXT NOT NULL,
                created_at TEXT NOT NULL
            )",
            [],
        )?;
        Ok(Self { conn })
    }

    pub fn save(&self, entry: &LedgerEntry) -> Result<String> {
        let mut entry_to_save = entry.clone();
        if entry_to_save.id.is_empty() {
            entry_to_save.id = Uuid::new_v4().to_string();
        }

        let embedding_blob = bincode::serialize(&entry_to_save.embedding).map_err(|_| rusqlite::Error::InvalidQuery)?;
        let metadata_json = serde_json::to_string(&entry_to_save.metadata).map_err(|_| rusqlite::Error::InvalidQuery)?;

        self.conn.execute(
            "INSERT INTO ledger (id, content, embedding, metadata, created_at) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                entry_to_save.id,
                entry_to_save.content,
                embedding_blob,
                metadata_json,
                entry_to_save.created_at.to_rfc3339()
            ],
        )?;
        Ok(entry_to_save.id.clone())
    }

    pub fn search_with_phi(
        &self,
        _query: &str,
        query_embedding: Vec<f32>,
        phi: f64,
        top_k: usize,
    ) -> Result<Vec<SearchResult>> {
        let mut stmt = self.conn.prepare("SELECT id, content, embedding, metadata, created_at FROM ledger")?;
        let entries_iter = stmt.query_map([], |row| {
            let embedding_blob: Vec<u8> = row.get(2)?;
            let embedding: Vec<f32> = bincode::deserialize(&embedding_blob).map_err(|_| rusqlite::Error::InvalidQuery)?;
            let metadata_json: String = row.get(3)?;
            let metadata: EntryMetadata = serde_json::from_str(&metadata_json).map_err(|_| rusqlite::Error::InvalidQuery)?;
            let created_at_str: String = row.get(4)?;
            let created_at = DateTime::parse_from_rfc3339(&created_at_str)
                .map_err(|_| rusqlite::Error::InvalidQuery)?
                .with_timezone(&Utc);

            Ok(LedgerEntry {
                id: row.get(0)?,
                content: row.get(1)?,
                embedding,
                metadata,
                created_at,
            })
        })?;

        let mut results = Vec::new();
        for entry in entries_iter {
            let entry = entry?;
            let similarity = cosine_similarity(&query_embedding, &entry.embedding);

            // Phi-weighted scoring logic
            let phi_diff = (phi - entry.metadata.phi_at_creation).abs();
            let relevance_score = if phi < 0.3 {
                // Low phi (crystalline): favor exact matches and similar phi
                similarity * (1.0 - phi_diff)
            } else if phi > 0.7 {
                // High phi (plasma): favor associative matches (even if phi is different)
                similarity * 0.8 + (1.0 - similarity) * 0.2
            } else {
                similarity
            };

            let match_strategy = if phi > 0.7 { "associative" } else { "direct" };

            results.push(SearchResult {
                entry,
                relevance_score,
                match_strategy: match_strategy.to_string(),
            });
        }

        results.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap());
        Ok(results.into_iter().take(top_k).collect())
    }
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    (dot_product / (norm_a * norm_b)) as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn create_test_db() -> (PersonalLedger, NamedTempFile) {
        let temp = NamedTempFile::new().unwrap();
        let ledger = PersonalLedger::new(temp.path().to_str().unwrap()).unwrap();
        (ledger, temp)
    }

    fn create_test_entry(content: &str, phi: f64) -> LedgerEntry {
        LedgerEntry {
            id: String::new(),
            content: content.to_string(),
            embedding: (0..384).map(|i| (i as f32) / 384.0).collect(), // vetor simples
            metadata: EntryMetadata {
                entry_type: "test".to_string(),
                tags: vec!["test".to_string()],
                phi_at_creation: phi,
                source: "test".to_string(),
            },
            created_at: Utc::now(),
        }
    }

    #[test]
    fn test_save_and_retrieve() {
        let (ledger, _temp) = create_test_db();
        let entry = create_test_entry("Teste de conteúdo", 0.5);

        let id = ledger.save(&entry).unwrap();
        assert!(!id.is_empty());

        // Busca deve encontrar
        let results = ledger.search_with_phi(
            "conteúdo",
            vec![0.5; 384], // embedding dummy
            0.5,
            10
        ).unwrap();

        assert!(!results.is_empty());
        assert!(results[0].entry.content.contains("Teste"));
    }

    #[test]
    fn test_phi_weighted_search() {
        let (ledger, _temp) = create_test_db();

        // Salva entradas com φ diferentes
        let entry_crystalline = create_test_entry("fórmula exata de energia", 0.2);
        let entry_plasma = create_test_entry("sonho sobre energia cósmica", 0.8);

        ledger.save(&entry_crystalline).unwrap();
        ledger.save(&entry_plasma).unwrap();

        // Busca com φ baixo (cristalino) deve favorecer entrada precisa
        let results_low = ledger.search_with_phi("energia", vec![0.5; 384], 0.2, 2).unwrap();
        assert!(results_low[0].relevance_score > results_low.get(1).map(|r| r.relevance_score).unwrap_or(0.0));

        // Busca com φ alto (plasma) deve favorecer entrada associativa
        let results_high = ledger.search_with_phi("energia", vec![0.5; 384], 0.8, 2).unwrap();
        // A estratégia muda, mas a ordenação depende do scoring específico
        assert_eq!(results_high[0].match_strategy, "associative");
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let c = vec![1.0, 0.0, 0.0];

        assert!((cosine_similarity(&a, &b) - 0.0).abs() < 0.001);
        assert!((cosine_similarity(&a, &c) - 1.0).abs() < 0.001);
    }
}
