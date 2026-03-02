// rust/src/checkpoint/scrubber.rs
use crate::error::ResilientResult;
use crate::state::ResilientState;

#[derive(Default)]
pub struct StateScrubber;

impl StateScrubber {
    pub fn scrub(&self, mut state: ResilientState) -> ResilientResult<ResilientState> {
        state.memory.summary = self.scrub_text(&state.memory.summary);
        Ok(state)
    }

    fn scrub_text(&self, text: &str) -> String {
        text.replace("AKIA", "[REDACTED]")
            .replace("private_key", "[REDACTED]")
    }
}
