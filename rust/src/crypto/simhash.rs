use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub struct SimHash;

impl SimHash {
    pub fn calculate(text: &str) -> u64 {
        // Use 3-character shingles for more robust similarity on small texts
        let shingles: Vec<String> = text.chars()
            .collect::<Vec<char>>()
            .windows(3)
            .map(|w| w.iter().collect())
            .collect();

        let mut v = vec![0i32; 64];

        if shingles.is_empty() {
            // Fallback to whitespace tokens if text is too short
            let tokens: Vec<&str> = text.split_whitespace().collect();
            for token in tokens {
                Self::update_vector(&mut v, token);
            }
        } else {
            for shingle in shingles {
                Self::update_vector(&mut v, &shingle);
            }
        }

        let mut fingerprint = 0u64;
        for i in 0..64 {
            if v[i] > 0 {
                fingerprint |= 1 << i;
            }
        }
        fingerprint
    }

    fn update_vector(v: &mut Vec<i32>, token: &str) {
        let hash = Self::hash_token(token);
        for i in 0..64 {
            if (hash >> i) & 1 == 1 {
                v[i] += 1;
            } else {
                v[i] -= 1;
            }
        }
    }

    fn hash_token(token: &str) -> u64 {
        let mut s = DefaultHasher::new();
        token.hash(&mut s);
        s.finish()
    }

    pub fn hamming_distance(h1: u64, h2: u64) -> u32 {
        (h1 ^ h2).count_ones()
    }

    pub fn are_similar(h1: u64, h2: u64, threshold: u32) -> bool {
        Self::hamming_distance(h1, h2) <= threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simhash_similarity() {
        let text1 = "This is a viral message about sovereignty.";
        let text2 = "This is a viral message about sovereignty! [Shared]"; // Modified
        let text3 = "The quick brown fox jumps over the lazy dog.";

        let h1 = SimHash::calculate(text1);
        let h2 = SimHash::calculate(text2);
        let h3 = SimHash::calculate(text3);

        let d12 = SimHash::hamming_distance(h1, h2);
        let d13 = SimHash::hamming_distance(h1, h3);

        assert!(d12 < d13);
        assert!(SimHash::are_similar(h1, h2, 15)); // Higher threshold for small variations
        assert!(!SimHash::are_similar(h1, h3, 15));
    }
}
