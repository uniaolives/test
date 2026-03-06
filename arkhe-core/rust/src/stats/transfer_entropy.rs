use std::collections::HashMap;

pub struct TransferEntropyEstimator {
    pub bins: usize,
    pub history_x: Vec<f64>,
    pub history_y: Vec<f64>,
    pub capacity: usize,
}

impl TransferEntropyEstimator {
    pub fn new(bins: usize, capacity: usize) -> Self {
        Self {
            bins,
            history_x: Vec::with_capacity(capacity),
            history_y: Vec::with_capacity(capacity),
            capacity,
        }
    }

    pub fn add_observation(&mut self, x: f64, y: f64) {
        if self.history_x.len() >= self.capacity {
            self.history_x.remove(0);
            self.history_y.remove(0);
        }
        self.history_x.push(x);
        self.history_y.push(y);
    }

    fn discretize(&self, value: f64) -> usize {
        let b = (value * self.bins as f64).floor() as usize;
        if b >= self.bins { self.bins - 1 } else { b }
    }

    pub fn calculate_te_x_to_y(&self) -> f64 {
        if self.history_x.len() < 2 {
            return 0.0;
        }

        let n = self.history_x.len() - 1;
        let mut joint_y1_y0_x0 = HashMap::new();
        let mut joint_y1_y0 = HashMap::new();
        let mut joint_y0_x0 = HashMap::new();
        let mut joint_y0 = HashMap::new();

        for t in 0..n {
            let x0 = self.discretize(self.history_x[t]);
            let y0 = self.discretize(self.history_y[t]);
            let y1 = self.discretize(self.history_y[t + 1]);

            *joint_y1_y0_x0.entry((y1, y0, x0)).or_insert(0.0) += 1.0;
            *joint_y1_y0.entry((y1, y0)).or_insert(0.0) += 1.0;
            *joint_y0_x0.entry((y0, x0)).or_insert(0.0) += 1.0;
            *joint_y0.entry(y0).or_insert(0.0) += 1.0;
        }

        let mut te = 0.0;
        let n_f = n as f64;

        for (&(y1, y0, x0), &count) in joint_y1_y0_x0.iter() {
            let p_y1_y0_x0 = count / n_f;
            let p_y0_x0 = joint_y0_x0.get(&(y0, x0)).cloned().unwrap_or(0.0) / n_f;
            let p_y1_y0 = joint_y1_y0.get(&(y1, y0)).cloned().unwrap_or(0.0) / n_f;
            let p_y0 = joint_y0.get(&y0).cloned().unwrap_or(0.0) / n_f;

            if p_y0_x0 > 0.0 && p_y1_y0 > 0.0 && p_y0 > 0.0 {
                let term = (p_y1_y0_x0 / p_y0_x0) / (p_y1_y0 / p_y0);
                if term > 0.0 {
                    te += p_y1_y0_x0 * term.log2();
                }
            }
        }

        te
    }
}
