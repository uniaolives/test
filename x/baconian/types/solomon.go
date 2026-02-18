package types

import (
	"cosmossdk.io/math"
	sdk "github.com/cosmos/cosmos-sdk/types"
)

type HouseOfSolomon struct {
	Members      []string    // endereços dos validadores (sábios)
	Experiments  []string    // IDs de experimentos em andamento
	Publications []string    // IDs de leis aprovadas
	Secrets      []string    // conhecimentos sensíveis (criptografados)
	Treasury     sdk.Coins   // fundos para pesquisa
}

// CanPropose: apenas membros podem propor induções
func (h HouseOfSolomon) CanPropose(member string) bool {
	for _, m := range h.Members {
		if m == member {
			return true
		}
	}
	return false
}

// ShouldReveal: revelação gradual: conhecimento é liberado conforme C_total aumenta
func (h HouseOfSolomon) ShouldReveal(secret string, globalCoherence math.LegacyDec, threshold math.LegacyDec) bool {
	return globalCoherence.GTE(threshold)
}
