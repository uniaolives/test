pub fn selector(name: &str) -> String {
    let hash = blake3::hash(name.as_bytes());
    format!("0x{}", hex::encode(&hash.as_bytes()[..4]))
}
