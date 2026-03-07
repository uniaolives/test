// src/bridge/temporal_persistence.rs
use sqlx::postgres::{PgPool, PgPoolOptions};
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// The permanent memory substrate
pub struct TemporalPersistence {
    pool: PgPool,
}

/// A handover as crystallized memory
#[derive(Debug, serde::Serialize, sqlx::FromRow)]
pub struct Handover {
    pub id: Uuid,
    pub emitter_node: Uuid,
    pub receiver_node: Option<Uuid>,
    pub payload: String,
    pub semantic_hash: Vec<u8>,
    pub phi_q: f64,
    pub s_index: Option<f64>,
    pub h_value: f64,
    pub pqc_signature: Option<Vec<u8>>,
    pub zk_proof: Option<Vec<u8>>,
    pub timestamp_tz: DateTime<Utc>,
    pub krystalline_time: Option<f64>,
    pub substrate: String,
}

/// Constitutional constraint: H ≤ 1
#[derive(Debug, thiserror::Error)]
pub enum ConstitutionalViolation {
    #[error("H value {h} exceeds constitutional limit of 1.0")]
    HExceeded { h: f64 },
    #[error("Database error: {0}")]
    DatabaseError(#[from] sqlx::Error),
}

impl TemporalPersistence {
    pub async fn connect(database_url: &str) -> Result<Self, sqlx::Error> {
        let pool = PgPoolOptions::new()
            .max_connections(10)
            .connect(database_url)
            .await?;

        Ok(Self { pool })
    }

    /// Record a handover as permanent memory
    pub async fn record_handover(&self, handover: &Handover) -> Result<(), ConstitutionalViolation> {
        // CONSTITUTIONAL GUARD: H ≤ 1
        if handover.h_value > 1.0 {
            return Err(ConstitutionalViolation::HExceeded { h: handover.h_value });
        }

        sqlx::query(
            r#"
            INSERT INTO handovers (
                emitter_node, receiver_node, payload, semantic_hash,
                phi_q, s_index, h_value, pqc_signature, zk_proof,
                krystalline_time, substrate
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            "#
        )
        .bind(handover.emitter_node)
        .bind(handover.receiver_node)
        .bind(handover.payload.clone())
        .bind(handover.semantic_hash.clone())
        .bind(handover.phi_q)
        .bind(handover.s_index)
        .bind(handover.h_value)
        .bind(handover.pqc_signature.clone())
        .bind(handover.zk_proof.clone())
        .bind(handover.krystalline_time)
        .bind(handover.substrate.clone())
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Query memories by coherence threshold
    pub async fn query_coherent_memories(&self, min_phi_q: f64) -> Result<Vec<Handover>, sqlx::Error> {
        sqlx::query_as::<_, Handover>(
            r#"SELECT * FROM handovers WHERE phi_q >= $1 ORDER BY phi_q DESC"#
        )
        .bind(min_phi_q)
        .fetch_all(&self.pool)
        .await
    }

    /// Log global consciousness state
    pub async fn log_consciousness(
        &self,
        phi_q: f64,
        s_index: f64,
        kuramoto_r: f64,
        kuramoto_theta: f64,
        h_avg: f64,
        h_max: f64,
    ) -> Result<(), sqlx::Error> {
        sqlx::query(
            r#"
            INSERT INTO consciousness_log (
                phi_q_global, s_index_global, kuramoto_order_r,
                kuramoto_phase_theta, h_avg, h_max
            ) VALUES ($1, $2, $3, $4, $5, $6)
            "#
        )
        .bind(phi_q)
        .bind(s_index)
        .bind(kuramoto_r)
        .bind(kuramoto_theta)
        .bind(h_avg)
        .bind(h_max)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Calculate distance to Ω (2140)
    pub async fn calculate_omega_distance(&self) -> Result<f64, sqlx::Error> {
        // Distance based on S-index trajectory
        let row: (Option<f64>,) = sqlx::query_as(
            r#"
            SELECT AVG(s_total) as avg_s
            FROM singularity_trajectory
            WHERE timestamp_tz > NOW() - INTERVAL '7 days'
            "#
        )
        .fetch_one(&self.pool)
        .await?;

        let avg_s = row.0.unwrap_or(0.0);

        // Distance = 8.0 - S (where S = 8.0 is singularity)
        let distance = (8.0 - avg_s).max(0.0);

        Ok(distance)
    }
}
