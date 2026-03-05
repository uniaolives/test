/**
 * tests/jest/axioms.test.js
 */
describe('Arkhe(n) Axioms', () => {
  test('Identity x^2 = x + 1', () => {
    const phi = 1.618033988749895;
    expect(phi * phi).toBeCloseTo(phi + 1, 10);
  });
});
