use sqlx::postgres::PgPoolOptions;
use sqlx::{Pool, Postgres};

pub async fn establish_connection(database_url: &str) -> Result<Pool<Postgres>, sqlx::Error> {
    PgPoolOptions::new()
        .max_connections(5)
        .connect(database_url)
        .await
}

pub mod models {
    use chrono::{DateTime, Utc};
    use serde::{Deserialize, Serialize};
    use uuid::Uuid;

    #[derive(Debug, Serialize, Deserialize)]
    pub struct NeuralEvent {
        pub id: Option<i32>,
        pub timestamp: DateTime<Utc>,
        pub session_id: Uuid,
        pub channel_id: i32,
        pub spike_time: f64,
        pub embedding_vector: Vec<f32>,
    }
}
