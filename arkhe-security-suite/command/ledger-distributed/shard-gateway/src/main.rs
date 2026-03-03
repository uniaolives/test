pub fn route_handover(id: &[u8], num_shards: usize) -> usize {
    let mut hash = 0u64;
    for &b in id {
        hash = hash.wrapping_add(b as u64);
    }
    (hash % num_shards as u64) as usize
}

fn main() {
    println!("Shard Gateway starting...");
}
