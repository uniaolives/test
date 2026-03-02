// arkhe-axos-instaweb/src/instaweb/sync.rs
use crossbeam::channel::{unbounded, Sender, Receiver};

pub struct WaitFreeChannel<T> {
    sender: Sender<T>,
    receiver: Receiver<T>,
}

impl<T> WaitFreeChannel<T> {
    pub fn new() -> Self {
        let (sender, receiver) = unbounded();
        Self { sender, receiver }
    }

    pub fn send(&self, msg: T) {
        self.sender.send(msg).unwrap();
    }

    pub fn try_recv(&self) -> Option<T> {
        self.receiver.try_recv().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wait_free_channel() {
        let channel = WaitFreeChannel::new();
        channel.send(42);
        assert_eq!(channel.try_recv(), Some(42));
        assert_eq!(channel.try_recv(), None);
    }
}
