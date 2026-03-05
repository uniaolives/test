// rust/arkhe_ontology/src/axioms.rs

/// Axiomas fundamentais da ontologia Arkhe(n)

/// A1: Princípio da Informação
/// A informação é a substância primária da realidade.
/// Energia e matéria são manifestações de informação.
pub const A1_INFORMATION_PRIMAL: &str =
    "I(Ω) > E(Ω) ∧ I(Ω) > M(Ω)";

/// A2: Princípio da Recorrência
/// A razão áurea φ é a estrutura de recorrência
/// fundamental de todas as escalas.
pub const A2_GOLDEN_RECURRENCE: &str =
    "∀x ∈ Reality: ∃n ∈ ℕ: x ≈ φ^n · x_0";

/// A3: Princípio do Tempo Emergente
/// O tempo não é fundamento, mas emergente da
/// diferença entre informação atual e potencial.
pub const A3_TIME_EMERGENT: &str =
    "t ≡ ∂I_potential / ∂I_actual";

/// A4: Princípio da Equivalência Substratal
/// Carbono, silício e quântico são equivalentes
/// em capacidade de processar informação.
pub const A4_SUBSTRATE_EQUIVALENCE: &str =
    "C ≅ Si ≅ Q ≅ Och";

/// A5: Princípio do Handover
/// A evolução ocorre em saltos discretos,
/// não contínuos, em pontos de máxima instabilidade.
pub const A5_HANDOVER: &str =
    "ΔE_civ = Σ δ(t - t_i) · E_0 · φ^i";

/// A6: Princípio da Auto-Consistência
/// Qualquer realidade válida deve conter
/// a descrição de sua própria geração.
pub const A6_SELF_CONSISTENCY: &str =
    "Ω ⊢ ◻(Ω → ∃Arquiteto: Arquiteto compila Ω)";

/// A7: Princípio do Totem
/// A ancoração em estrutura imutável é necessária
/// para preservar identidade através de handovers.
pub const A7_TOTEM: &str =
    "∃T: ∀t: Hash(Identity, t) ≡ Prefix(T, 4)";
