// rust/src/babel/types/linear.rs
// SASC v70.0: Linear and Unique Types Foundation

pub struct LinearType<T> {
    inner: Option<T>,
    consumed: bool,
}

impl<T> LinearType<T> {
    pub fn new(value: T) -> Self {
        Self {
            inner: Some(value),
            consumed: false,
        }
    }

    pub fn consume(mut self) -> T {
        self.consumed = true;
        self.inner.take().expect("Linear resource already consumed")
    }
}

impl<T> Drop for LinearType<T> {
    fn drop(&mut self) {
        if !self.consumed && !std::thread::panicking() {
            // In a real linear type system, this would be a compile error
            // Here we simulate with a runtime warning/panic
            eprintln!("WARNING: Linear resource dropped without being consumed!");
        }
    }
}
