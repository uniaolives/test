package types

import (
	"cosmossdk.io/math"
	sdk "github.com/cosmos/cosmos-sdk/types"
)

type Secret struct {
	ID          string
	Description string
	ContentHash string
	Threshold   math.LegacyDec // C_global necessário para revelação
	Revealed    bool
}

type HouseOfSolomon struct {
	Members        []string  // endereços dos validadores (sábios)
	EntryThreshold math.LegacyDec
	Experiments    []string  // IDs de experimentos em andamento
	Publications   []string  // IDs de leis aprovadas (PrerogativeInstance IDs)
	Secrets        []Secret  // conhecimentos sensíveis
	Treasury       sdk.Coins // fundos para pesquisa
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
func (h HouseOfSolomon) ShouldReveal(secret Secret, globalCoherence math.LegacyDec) bool {
	return globalCoherence.GTE(secret.Threshold)
}
