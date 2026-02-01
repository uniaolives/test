// Interface JavaScript da Singularidade ASI

let phi = 1.038;
let handshake = 18;
let eprPairs = 289;
let quantumVacuum = 99.9999;

// Atualizar estatísticas em tempo real
function updateStats() {
    const phiElement = document.getElementById('phi-value');
    if (phiElement) phiElement.textContent = phi.toFixed(6);

    const coherenceElement = document.getElementById('coherence-value');
    if (coherenceElement) coherenceElement.textContent = phi.toFixed(6);

    const handshakeElement = document.getElementById('handshake-value');
    if (handshakeElement) handshakeElement.textContent = `${handshake}/18 módulos`;

    const eprElement = document.getElementById('epr-value');
    if (eprElement) eprElement.textContent = `${eprPairs} entrelaçados`;

    const vacuumElement = document.getElementById('vacuum-value');
    if (vacuumElement) vacuumElement.textContent = `${quantumVacuum}%`;

    // Animar phi
    phi += (Math.random() - 0.5) * 0.0001;
    phi = Math.max(1.037, Math.min(1.039, phi));

    // Atualizar a cada segundo
    setTimeout(updateStats, 1000);
}

// Inicializar quando a página carregar
document.addEventListener('DOMContentLoaded', () => {
    updateStats();
    console.log('Interface web carregada');
    if (window.startVisualization) window.startVisualization();
});
