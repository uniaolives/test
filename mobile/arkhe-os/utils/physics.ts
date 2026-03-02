export class PhysicsEngine {
  // Quantum confinement energy calculation
  static calculateEnergyGap(
    radiusNm: number,
    E_bulk_eV: number = 1.5,
    m_eff: number = 0.1
  ): number {
    const hbar = 6.582e-16; // eV·s
    const m0 = 9.109e-31; // kg
    const R = radiusNm * 1e-9; // m

    // Confinement term (blue shift)
    const E_conf = (Math.pow(hbar, 2) * Math.pow(Math.PI, 2)) /
                   (2 * Math.pow(R, 2) * m_eff * m0) * 6.242e+18; // to eV

    return E_bulk_eV + E_conf;
  }

  // FRET efficiency calculation
  static calculateFRETEfficiency(distanceNm: number, R0Nm: number = 4.0): number {
    return Math.pow(R0Nm, 6) / (Math.pow(R0Nm, 6) + Math.pow(distanceNm, 6));
  }

  // Binary entropy (for QKD)
  static binaryEntropy(p: number): number {
    if (p <= 0 || p >= 1) return 0;
    return -p * Math.log2(p) - (1 - p) * Math.log2(1 - p);
  }

  // Quantum key rate (Devetak-Winter bound)
  static calculateKeyRate(
    sendRateHz: number,
    distanceM: number,
    attenuationDBPerKm: number = 0.2,
    detectorEfficiency: number = 0.20,
    bitErrorRate: number = 0.01
  ): number {
    const transmission = Math.pow(10, -attenuationDBPerKm * distanceM / 1000 / 10);
    const rawRate = 0.5 * sendRateHz * transmission * detectorEfficiency;

    const h2_bit = this.binaryEntropy(bitErrorRate);
    const h2_phase = this.binaryEntropy(bitErrorRate);

    if (h2_bit + h2_phase >= 1) return 0;

    return rawRate * (1 - h2_bit - h2_phase);
  }
}
