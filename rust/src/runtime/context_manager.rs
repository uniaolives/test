// rust/src/runtime/context_manager.rs

pub struct Message {
    pub is_user: bool,
    pub content: String,
}

pub struct ContextWindow {
    pub messages: Vec<Message>,
    pub max_tokens: usize,
}

impl ContextWindow {
    pub fn new(max_tokens: usize) -> Self {
        Self { messages: Vec::new(), max_tokens }
    }
}

pub struct ContextManager;
