package types

import (
	"cosmossdk.io/math"
)

// Handover representa uma operação atômica de transferência de coerência no sistema meta-operacional.
type Handover struct {
	SourceID     string   `json:"source_id"`
	TargetID     string   `json:"target_id"`
	Payload      []byte   `json:"payload"`       // informação a transferir
	CoherenceIn  math.LegacyDec `json:"coherence_in"`  // coerência do nó fonte
	CoherenceOut math.LegacyDec `json:"coherence_out"` // coerência esperada no destino
	PhiRequired  math.LegacyDec `json:"phi_required"`  // integração mínima necessária
	TTL          int      `json:"ttl"`           // tempo de vida em handovers
}

// NewHandover cria uma nova instância de Handover com os parâmetros validados.
func NewHandover(source, target string, payload []byte, cIn, cOut, phi math.LegacyDec, ttl int) Handover {
	return Handover{
		SourceID:     source,
		TargetID:     target,
		Payload:      payload,
		CoherenceIn:  cIn,
		CoherenceOut: cOut,
		PhiRequired:  phi,
		TTL:          ttl,
	}
}
