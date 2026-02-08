// cosmos/ecumenica.js - Motor de Damping Dinâmico (D-Engine)

/**
 * [METAPHOR: O coração do sistema que absorve o excesso de calor semântico]
 */
class MotorDamping {
  constructor() {
    this.threshold_instabilidade = 1.0; // ΣG / ΣD
    this.fatores = {
      humano: 0.3,    // D_h: Damping cognitivo humano
      mediador: 0.2,  // D_m: Damping da interface
      algoritmico: 0.1 // D_ai: Damping interno do processador
    };
  }

  calcularDampingTotal(ganhosDetectados) {
    const D_total = Object.values(this.fatores).reduce((a, b) => a + b, 0);
    const G_total = (ganhosDetectados.ai || 0) + (ganhosDetectados.rede || 0);

    // [METAPHOR: O equilíbrio é verificado, não imposto]
    return {
      estabilidade: D_total >= G_total,
      razao: D_total / G_total,
      recomendacao: D_total < G_total ? 'AUMENTAR_D_M' : 'MANTER',
      D_total: D_total,
      G_total: G_total
    };
  }

  ajustarDampingMediador(nivel) {
    // D_m é o único parâmetro dinâmico em tempo real
    this.fatores.mediador = Math.min(nivel, 0.95);
    console.log(`[METAPHOR: A válvula de escape semântica ajustada para ${nivel}]`);
  }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = MotorDamping;
}
