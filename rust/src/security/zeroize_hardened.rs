// src/security/zeroize_hardened.rs
// ORDEM 3: Mandate Zeroize-on-drop para TODOS os buffers sensíveis

pub use zeroize::{Zeroize, ZeroizeOnDrop};
use std::ops::{Deref, DerefMut};

/// Wrapper que garante zeroização imediata ao sair de escopo
#[derive(Zeroize, ZeroizeOnDrop)]
pub struct HardenedBuffer<T: Zeroize> {
    inner: T,
    #[zeroize(skip)] // Não limpar metadado de tamanho
    pub size: usize,
}

impl<T: Zeroize + Default> HardenedBuffer<T> {
    pub fn new(data: T) -> Self {
        let size = std::mem::size_of_val(&data);
        Self { inner: data, size }
    }
}

impl<T: Zeroize> Deref for HardenedBuffer<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T: Zeroize> DerefMut for HardenedBuffer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}
