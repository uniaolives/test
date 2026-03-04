describe('Arkhe(n) E2E', () => {
  it('Verifies global coherence sync', () => {
    cy.visit('/');
    cy.get('[data-cy=coherence]').should('exist');
  });
});
