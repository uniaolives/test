use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use arkhe_quantum::Handover;

pub struct ShardRouter {
    pub num_shards: u64,
}

impl ShardRouter {
    pub fn new(num_shards: u64) -> Self {
        Self { num_shards }
    }

    pub fn get_shard(&self, handover_id: &uuid::Uuid) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        handover_id.hash(&mut hasher);
        hasher.finish() % self.num_shards
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn test_shard_routing() {
        let router = ShardRouter::new(4);
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        let s1 = router.get_shard(&id1);
        let s2 = router.get_shard(&id2);

        assert!(s1 < 4);
        assert!(s2 < 4);
    }
}

pub struct ShardNode {
    pub id: u64,
    pub storage: Vec<Handover>,
}

impl ShardNode {
    pub fn new(id: u64) -> Self {
        Self { id, storage: Vec::new() }
    }

    pub fn append(&mut self, handover: Handover) {
        self.storage.push(handover);
    }
}

pub fn main() {
    println!("Arkhe(n) Shard Module");
}
