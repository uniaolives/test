// arkhe_omni_system/consensus/criticality.go
package consensus

import (
	"math"
	"math/rand"
)

// CriticalityConsensus substitui votação por emergência termodinâmica.
type CriticalityConsensus struct {
	Temperature float64 // Controle de ruído (Beta = 1/T)
	Threshold   float64 // Limiar Phi (0.618)
}

// NodeState representa a opinião/voto de um nó como um spin Ising (+1 ou -1).
type NodeState int

const (
	ACCEPT NodeState = 1
	REJECT NodeState = -1
)

// EvaluateConsensus simula a emergência de um estado global através de interações locais.
func (c *CriticalityConsensus) EvaluateConsensus(networkStates []NodeState) (NodeState, float64) {
	if len(networkStates) == 0 {
		return REJECT, 0.0
	}

	// Calcula a magnetização média (m = sum(si) / N)
	var sum float64
	for _, s := range networkStates {
		sum += float64(s)
	}
	magnetization := sum / float64(len(networkStates))

	// Calcula a criticidade baseada na flutuação (Susceptibilidade)
	// Em sistemas críticos, a flutuação é máxima no ponto de transição.
	phi := 1.0 - math.Abs(magnetization-0.618) // Proximidade do ponto crítico Arkhe

	// Decisão probabilística baseada na temperatura (Boltzmann)
	energy := -magnetization
	prob := 1.0 / (1.0 + math.Exp(energy/c.Temperature))

	if rand.Float64() < prob && phi >= c.Threshold {
		return ACCEPT, phi
	}
	return REJECT, phi
}

// AdjustTemperature ajusta a temperatura do sistema para manter a criticidade phi.
func (c *CriticalityConsensus) AdjustTemperature(currentPhi float64) {
	if currentPhi < c.Threshold {
		c.Temperature *= 1.05 // Aumenta ruído para evitar congelamento subcrítico
	} else {
		c.Temperature *= 0.95 // Reduz ruído para estabilizar o ponto crítico
	}
}
