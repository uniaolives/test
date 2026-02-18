package keeper

import (
	"strings"
	"testing"

	cosmosmath "cosmossdk.io/math"
	sdk "github.com/cosmos/cosmos-sdk/types"
	"arkhend/x/baconian/types"
)

func TestDetectIdols(t *testing.T) {
	k := Keeper{}

	// Caso 1: Idolum Theatri (Viés de confirmação)
	observations := []types.Observation{
		{Context: "wood oxygen spark", Result: true, Intensity: cosmosmath.LegacyNewDec(1), Validator: "v1"},
		{Context: "paper oxygen flame", Result: true, Intensity: cosmosmath.LegacyNewDec(1), Validator: "v2"},
		{Context: "gas oxygen spark", Result: true, Intensity: cosmosmath.LegacyNewDec(1), Validator: "v3"},
		{Context: "wood oxygen spark", Result: true, Intensity: cosmosmath.LegacyNewDec(1), Validator: "v1"},
		{Context: "paper oxygen flame", Result: true, Intensity: cosmosmath.LegacyNewDec(1), Validator: "v2"},
		{Context: "gas oxygen spark", Result: true, Intensity: cosmosmath.LegacyNewDec(1), Validator: "v3"},
		{Context: "wood oxygen spark", Result: true, Intensity: cosmosmath.LegacyNewDec(1), Validator: "v1"},
		{Context: "paper oxygen flame", Result: true, Intensity: cosmosmath.LegacyNewDec(1), Validator: "v2"},
		{Context: "gas oxygen spark", Result: true, Intensity: cosmosmath.LegacyNewDec(1), Validator: "v3"},
		{Context: "wood oxygen spark", Result: true, Intensity: cosmosmath.LegacyNewDec(1), Validator: "v1"},
	}

	table := types.Table{
		Phenomenon:   "combustion",
		Presence:     observations,
		Absence:      []types.Observation{},
		ValidatorSet: []string{"v1", "v2", "v3"},
	}

	idols := k.DetectIdols(sdk.Context{}, table)
	foundTheatri := false
	for _, idol := range idols {
		if idol.Type == types.IdolumTheatri {
			foundTheatri = true
		}
	}
	if !foundTheatri {
		t.Error("Esperado Idolum Theatri devido à falta de instâncias de ausência")
	}

	// Caso 2: Idolum Tribus (Baixa diversidade)
	table.ValidatorSet = []string{"v1"}
	// Adiciona mais observações para passar de 20
	for i := 0; i < 15; i++ {
		observations = append(observations, types.Observation{Result: true, Validator: "v1"})
	}
	table.Presence = observations
	idols = k.DetectIdols(sdk.Context{}, table)
	foundTribus := false
	for _, idol := range idols {
		if idol.Type == types.IdolumTribus {
			foundTribus = true
		}
	}
	if !foundTribus {
		t.Error("Esperado Idolum Tribus devido à baixa diversidade de validadores")
	}
}

func TestInferLaw(t *testing.T) {
	k := Keeper{}

	presence := []types.Observation{
		{Context: "wood oxygen spark", Result: true},
		{Context: "paper oxygen flame", Result: true},
	}
	absence := []types.Observation{
		{Context: "wood nitrogen spark", Result: false},
	}

	table := types.Table{
		Phenomenon: "combustion",
		Presence:   presence,
		Absence:    absence,
	}

	law, confidence, _, _ := k.InferLaw(table)

	if !contains(law, "oxygen") {
		t.Errorf("Lei esperada conter 'oxygen', got: %s", law)
	}

	if confidence.IsZero() {
		t.Error("Confiança não deve ser zero para dados consistentes")
	}
}

func contains(s string, substr string) bool {
	return strings.Contains(s, substr)
}
