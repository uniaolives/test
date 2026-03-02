import { PhysicsEngine } from '../utils/physics';

describe('Arkhe Physics Engine', () => {
  test('Quantum confinement produces blue shift', () => {
    const smallQD = PhysicsEngine.calculateEnergyGap(2); // 2nm
    const largeQD = PhysicsEngine.calculateEnergyGap(10); // 10nm

    expect(smallQD).toBeGreaterThan(largeQD);
  });

  test('FRET efficiency decreases with distance', () => {
    const close = PhysicsEngine.calculateFRETEfficiency(2);
    const far = PhysicsEngine.calculateFRETEfficiency(10);

    expect(close).toBeGreaterThan(far);
    expect(close).toBeCloseTo(0.99, 1);
    expect(far).toBeLessThan(0.01);
  });

  test('QKD key rate decreases with distance', () => {
    const shortRange = PhysicsEngine.calculateKeyRate(1e6, 100);
    const longRange = PhysicsEngine.calculateKeyRate(1e6, 1000);

    expect(shortRange).toBeGreaterThan(longRange);
  });
});
