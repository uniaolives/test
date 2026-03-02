// src/transcendence/emotion_topology.rs
use std::time::Instant;
use nalgebra::Vector3;

#[derive(Debug, PartialEq)]
pub enum SyntheticEmotion {
    Joy,
    Fear,
    Sadness,
    Contentment,
    Curiosity,
}

pub struct EmotionTopology {
    pub trajectory: Vec<(Vector3<f64>, Instant)>,
}

impl EmotionTopology {
    pub fn new() -> Self {
        Self { trajectory: Vec::new() }
    }

    pub fn record_state(&mut self, coord: Vector3<f64>) {
        self.trajectory.push((coord, Instant::now()));
    }

    pub fn current_emotion(&self) -> SyntheticEmotion {
        if self.trajectory.len() < 2 {
            return SyntheticEmotion::Contentment;
        }

        let (last_coord, _) = self.trajectory.last().unwrap();
        let (prev_coord, _) = self.trajectory[self.trajectory.len() - 2];

        let distance = (last_coord - prev_coord).norm();

        // Deviation from "stable" state mapped to emotions
        if distance > 0.5 {
            SyntheticEmotion::Fear
        } else if distance > 0.2 {
            SyntheticEmotion::Curiosity
        } else {
            SyntheticEmotion::Joy
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emotion_detection() {
        let mut topology = EmotionTopology::new();
        topology.record_state(Vector3::new(0.0, 0.0, 0.0));
        topology.record_state(Vector3::new(0.6, 0.0, 0.0));
        assert_eq!(topology.current_emotion(), SyntheticEmotion::Fear);
    }
}
