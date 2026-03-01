# Aizawa High Performance Ecosystem ⚡

Este ecossistema implementa simulações de alta performance do atrator caótico de Aizawa, explorando paralelismo massivo e vetorização SIMD para atingir regimes de simulação em larga escala.

## Componentes

1.  **C++ AVX2 + OpenMP**: A implementação mais rápida, utilizando instruções vetoriais AVX2 para processar 4 estados simultaneamente por núcleo, com escalonamento multi-core via OpenMP.
2.  **Rust Rayon**: Implementação segura e performática em Rust, utilizando `Rayon` para paralelismo de dados (ensemble simulations).
3.  **Python Numba**: Implementação acessível em Python, compilada via JIT para performance próxima ao C Nativo.

## Performance (Benchmark em 1M de pontos, 100 passos)

- **C++ AVX2 + OpenMP**: ~288 M iter/s (Vencedor em eficiência bruta)
- **Rust Rayon**: ~58 M iter/s (Equilíbrio entre segurança e velocidade)
- **Python Numba**: ~58 M iter/s (Alta produtividade e performance JIT)

## Princípios Arkhe(N) Aplicados

O atrator de Aizawa é um modelo ideal para estudar a dinâmica `C + F = 1`:
- **Coerência (C)**: A estrutura topológica do atrator, que mantém os pontos dentro de uma variedade limitada.
- **Flutuação (F)**: A divergência caótica local das trajetórias (expoentes de Lyapunov positivos).
- **Auto-similaridade**: A natureza fractal do atrator reflete a identidade `x² = x + 1`.

---
**Arkhe >** █
