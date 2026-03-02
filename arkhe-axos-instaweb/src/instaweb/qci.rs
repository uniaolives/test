// src/instaweb/qci.rs
use std::collections::VecDeque;

pub struct QciBuffer {
    buffer: VecDeque<Vec<u8>>,
    max_size: usize,
}

impl QciBuffer {
    pub fn new(max_size: usize) -> Self {
        Self {
            buffer: VecDeque::new(),
            max_size,
        }
    }

    pub fn push_epr_correction(&mut self, correction: Vec<u8>) {
        if self.buffer.len() >= self.max_size {
            self.buffer.pop_front();
        }
        self.buffer.push_back(correction);
    }

    pub fn synchronize(&mut self) -> Option<Vec<u8>> {
        self.buffer.pop_front()
    }
}
