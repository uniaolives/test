pub struct TransferEntropy {
    pub history_x: Vec<f64>,
    pub history_y: Vec<f64>,
    pub capacity: usize,
}

impl TransferEntropy {
    pub fn new(capacity: usize) -> Self {
        Self {
            history_x: Vec::with_capacity(capacity),
            history_y: Vec::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, x: f64, y: f64) {
        self.history_x.push(x);
        self.history_y.push(y);
        if self.history_x.len() > self.capacity {
            self.history_x.remove(0);
            self.history_y.remove(0);
        }
    }

    pub fn calculate(&self) -> f64 {
        let n = self.history_x.len();
        if n < 10 { return 0.0; }

        // Simplificação: TE aproximada via correlação assimétrica defasada
        // T_{X->Y} = I(Y_{t+1} ; X_t | Y_t)
        let mut sum_te = 0.0;
        for i in 0..n-1 {
            let contribution = (self.history_y[i+1] - self.history_y[i]) * self.history_x[i];
            sum_te += contribution;
        }

        sum_te.abs() / n as f64
    }
}
