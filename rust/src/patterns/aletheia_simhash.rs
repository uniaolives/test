use crate::crypto::simhash::SimHash;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct TelegramMessage {
    pub id: String,
    pub group: String,
    pub author: String,
    pub text: String,
    pub timestamp: String,
}

pub struct AletheiaSimHashEngine {
    pub messages: Vec<TelegramMessage>,
}

impl AletheiaSimHashEngine {
    pub fn new() -> Self {
        Self { messages: Vec::new() }
    }

    pub fn add_message(&mut self, msg: TelegramMessage) {
        self.messages.push(msg);
    }

    pub fn cluster_messages(&self, threshold: u32) -> HashMap<u64, Vec<String>> {
        let mut clusters: HashMap<u64, Vec<String>> = HashMap::new();

        for msg in &self.messages {
            let fingerprint = SimHash::calculate(&msg.text);

            let mut found_cluster = None;
            for &existing_fp in clusters.keys() {
                if SimHash::are_similar(fingerprint, existing_fp, threshold) {
                    found_cluster = Some(existing_fp);
                    break;
                }
            }

            let cluster_id = found_cluster.unwrap_or(fingerprint);
            clusters.entry(cluster_id).or_insert(Vec::new()).push(msg.id.clone());
        }

        clusters
    }

    pub fn build_co_occurrence_network(&self, threshold: u32) -> Vec<(String, String, u32)> {
        let clusters = self.cluster_messages(threshold);
        let mut group_connections: HashMap<(String, String), u32> = HashMap::new();

        for (_fp, msg_ids) in clusters {
            let mut groups: Vec<String> = self.messages.iter()
                .filter(|m| msg_ids.contains(&m.id))
                .map(|m| m.group.clone())
                .collect();

            groups.sort();
            groups.dedup();

            // Create edges between groups that share a message (or similar messages)
            for i in 0..groups.len() {
                for j in i + 1..groups.len() {
                    let pair = (groups[i].clone(), groups[j].clone());
                    *group_connections.entry(pair).or_insert(0) += 1;
                }
            }
        }

        group_connections.into_iter()
            .map(|((g1, g2), weight)| (g1, g2, weight))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_clustering() {
        let mut engine = AletheiaSimHashEngine::new();
        engine.add_message(TelegramMessage {
            id: "1".to_string(),
            group: "G1".to_string(),
            author: "A1".to_string(),
            text: "This is a viral message about sovereignty.".to_string(),
            timestamp: "T1".to_string(),
        });
        engine.add_message(TelegramMessage {
            id: "2".to_string(),
            group: "G2".to_string(),
            author: "A2".to_string(),
            text: "This is a viral message about sovereignty! [Shared]".to_string(), // Similar
            timestamp: "T2".to_string(),
        });

        let clusters = engine.cluster_messages(15);
        assert_eq!(clusters.len(), 1);

        let network = engine.build_co_occurrence_network(15);
        assert_eq!(network.len(), 1);
        assert!(network[0].0 == "G1" || network[0].0 == "G2");
    }
}
