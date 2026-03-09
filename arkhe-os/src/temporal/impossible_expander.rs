use super::impossible_timeline::{ImpossibleTimeline, DivergentConstants, ImpossibilityClass, TransformedOrb};
use crate::propagation::payload::OrbPayload;
use uuid::Uuid;
use num_complex::Complex;

pub struct ImpossibleTimelineExpander {
    pub impossible_timelines: Vec<ImpossibleTimeline>,
}

impl ImpossibleTimelineExpander {
    pub fn new() -> Self {
        Self {
            impossible_timelines: Vec::new(),
        }
    }

    /// Generates standard impossible timelines
    pub fn generate_default(&mut self) {
        // Tachyonic Line
        self.impossible_timelines.push(ImpossibleTimeline {
            timeline_id: Uuid::new_v4(),
            constants: DivergentConstants {
                c: Complex::new(f64::INFINITY, 0.0),
                h_bar: 1.054e-34,
                g: 6.674e-11,
                lambda: 0.0,
                lambda_max: f64::INFINITY,
            },
            impossibility_class: ImpossibilityClass::TachyonicNorm,
            lambda_2: f64::INFINITY,
            tunnel_probability: 1.0,
        });

        // Reverse Entropy Line
        self.impossible_timelines.push(ImpossibleTimeline {
            timeline_id: Uuid::new_v4(),
            constants: DivergentConstants {
                c: Complex::new(299792458.0, 0.0),
                h_bar: -1.054e-34,
                g: 6.674e-11,
                lambda: 0.0,
                lambda_max: 1.5,
            },
            impossibility_class: ImpossibilityClass::ReverseEntropy,
            lambda_2: 1.5,
            tunnel_probability: 0.5,
        });

        // Macroscopic Quantum Line
        self.impossible_timelines.push(ImpossibleTimeline {
            timeline_id: Uuid::new_v4(),
            constants: DivergentConstants {
                c: Complex::new(299792458.0, 0.0),
                h_bar: 1.054e-34 * 1e30,
                g: 6.674e-11,
                lambda: 0.0,
                lambda_max: 1.0,
            },
            impossibility_class: ImpossibilityClass::MacroscopicQuantum,
            lambda_2: 1.0,
            tunnel_probability: 0.9,
        });
    }

    pub fn find_best_timeline(&self, orb: &OrbPayload) -> Option<ImpossibleTimeline> {
        self.impossible_timelines
            .iter()
            .filter(|t| t.can_support_orb(orb))
            .max_by(|a, b| a.tunnel_probability.partial_cmp(&b.tunnel_probability).unwrap())
            .cloned()
    }

    pub fn tunnel_to_impossible(
        &self,
        orb: &OrbPayload,
        timeline: &ImpossibleTimeline,
    ) -> Result<TransformedOrb, &'static str> {
        if !timeline.can_support_orb(orb) {
            return Err("Incompatible orbital");
        }

        if timeline.tunnel_probability < 0.01 {
            return Err("Probability too low");
        }

        Ok(timeline.transform_orb(orb))
    }
}
