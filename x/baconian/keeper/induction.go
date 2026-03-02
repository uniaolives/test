package keeper

import (
	"fmt"
	"math"
	"strings"

	cosmosmath "cosmossdk.io/math"
	sdk "github.com/cosmos/cosmos-sdk/types"
	"arkhend/x/baconian/types"
)

type Keeper struct {
	// Em uma implementação real, incluiria keys de store e codec
}

func (k Keeper) PerformInduction(ctx sdk.Context, phenomenon string) (*types.PrerogativeInstance, error) {
	// 1. Recuperar todas as observações do fenômeno
	observations := k.GetObservationsByPhenomenon(ctx, phenomenon)
	if len(observations) < 10 {
		return nil, fmt.Errorf("observações insuficientes para indução (mínimo 10)")
	}

	// 2. Tabulação
	presence := filter(observations, func(o types.Observation) bool { return o.Result })
	absence := filter(observations, func(o types.Observation) bool { return !o.Result })
	degrees := filter(observations, func(o types.Observation) bool { return o.Intensity.GT(cosmosmath.LegacyZeroDec()) })

	table := types.Table{
		Phenomenon:   phenomenon,
		Presence:     presence,
		Absence:      absence,
		Degrees:      degrees,
		ValidatorSet: getUniqueValidators(observations),
	}

	// 3. Eliminatio (Detecção de Ídolos)
	idols := k.DetectIdols(ctx, table)

	// 4. Inductio (Inferência Causal)
	proposedLaw, confidence, support, specificity := k.InferLaw(table)

	// 5. Instância Privilegiada
	instance := types.PrerogativeInstance{
		ID:          fmt.Sprintf("IND-%s-%d", phenomenon, ctx.BlockHeight()),
		Phenomenon:  phenomenon,
		Table:       table,
		Idols:       idols,
		ProposedLaw: proposedLaw,
		Confidence:  confidence,
		Support:     support,
		Specificity: specificity,
		Status:      types.StatusDraft,
		Validator:   sdk.ConsAddress(ctx.BlockHeader().ProposerAddress).String(),
		BlockHeight: ctx.BlockHeight(),
	}

	// 6. Submeter à governança
	k.SubmitInductionProposal(ctx, instance)

	return &instance, nil
}

func (k Keeper) DetectIdols(ctx sdk.Context, table types.Table) []types.Idol {
	var idols []types.Idol
	allObs := append(table.Presence, table.Absence...)

	// Idolum Tribus: baixa diversidade de observadores
	if len(table.ValidatorSet) < 3 && len(allObs) > 20 {
		idols = append(idols, types.Idol{
			ID:          "IDOL-TRIBUS-" + table.Phenomenon,
			Type:        types.IdolumTribus,
			Severity:    cosmosmath.LegacyNewDecWithPrec(1, 0), // 1.0
			Description: "Baixa diversidade de observadores (Ídolo da Tribo)",
			Correction:  types.Correction{Strategy: "reweighted_penalty", Parameters: []string{"0.5"}},
		})
	}

	// Idolum Specus: outliers individuais via Z-Score
	intensities := getIntensities(allObs)
	mean, std := calculateStats(intensities)
	if std > 0 {
		for _, validator := range table.ValidatorSet {
			vObs := filterByNode(allObs, validator)
			vMean, _ := calculateStats(getIntensities(vObs))
			zScore := math.Abs(vMean-mean) / std
			if zScore > 2.0 {
				idols = append(idols, types.Idol{
					ID:            fmt.Sprintf("IDOL-SPECUS-%s", validator),
					Type:          types.IdolumSpecus,
					Severity:      cosmosmath.LegacyNewDecWithPrec(int64(zScore*10), 1),
					Description:   fmt.Sprintf("Viés individual detectado para o nó %s (Ídolo da Caverna)", validator),
					AffectedNodes: []string{validator},
				})
			}
		}
	}

	// Idolum Fori: alta variabilidade semântica (unique ratio)
	contexts := make(map[string]bool)
	for _, o := range allObs {
		contexts[o.Context] = true
	}
	uniqueRatio := float64(len(contexts)) / float64(len(allObs))
	if uniqueRatio > 0.5 {
		idols = append(idols, types.Idol{
			ID:          "IDOL-FORI-" + table.Phenomenon,
			Type:        types.IdolumFori,
			Severity:    cosmosmath.LegacyNewDecWithPrec(int64(uniqueRatio*100), 2),
			Description: "Confusão semântica nos contextos (Ídolo do Fórum)",
			Correction:  types.Correction{Strategy: "semantic_clustering", Parameters: []string{}},
		})
	}

	// Idolum Theatri: Confirmation Bias (presença excessiva)
	presenceRatio := float64(len(table.Presence)) / float64(len(allObs))
	if presenceRatio > 0.9 {
		idols = append(idols, types.Idol{
			ID:          "IDOL-THEATRI-" + table.Phenomenon,
			Type:        types.IdolumTheatri,
			Severity:    cosmosmath.LegacyNewDecWithPrec(int64(presenceRatio*100), 2),
			Description: "Viés de confirmação: falta de controles negativos (Ídolo do Teatro)",
			Correction:  types.Correction{Strategy: "require_absence", Parameters: []string{"10"}},
		})
	}

	return idols
}

func (k Keeper) InferLaw(table types.Table) (law string, confidence, support, specificity types.Dec) {
	// Algoritmo simplificado de indução baconiana
	if len(table.Presence) == 0 {
		return "Nenhuma lei inferida (sem instâncias positivas)", cosmosmath.LegacyZeroDec(), cosmosmath.LegacyZeroDec(), cosmosmath.LegacyZeroDec()
	}

	// 1. Extrair features comuns (interseção) dos contextos de presença
	commonFeatures := extractCommonFeatures(table.Presence)

	// 2. Verificar se essas features aparecem nos contextos de ausência (suficiência)
	sufficencyScore := 1.0
	for _, obs := range table.Absence {
		if containsAll(obs.Context, commonFeatures) {
			sufficencyScore -= 1.0 / float64(len(table.Absence)+1)
		}
	}

	// 3. Métricas
	totalCount := float64(len(table.Presence) + len(table.Absence))
	supportVal := float64(len(table.Presence)) / totalCount
	specificityVal := 1.0
	if len(table.Absence) > 0 {
		negativeMatches := 0
		for _, obs := range table.Absence {
			if containsAll(obs.Context, commonFeatures) {
				negativeMatches++
			}
		}
		specificityVal = 1.0 - (float64(negativeMatches) / float64(len(table.Absence)))
	}

	confidenceVal := supportVal * specificityVal * sufficencyScore
	law = fmt.Sprintf("∀x: presence(%s) → %s occurs", strings.Join(commonFeatures, ", "), table.Phenomenon)

	return law,
		cosmosmath.LegacyNewDecWithPrec(int64(confidenceVal*100), 2),
		cosmosmath.LegacyNewDecWithPrec(int64(supportVal*100), 2),
		cosmosmath.LegacyNewDecWithPrec(int64(specificityVal*100), 2)
}

// Helpers

func filter(obs []types.Observation, f func(types.Observation) bool) []types.Observation {
	var filtered []types.Observation
	for _, o := range obs {
		if f(o) {
			filtered = append(filtered, o)
		}
	}
	return filtered
}

func filterByNode(obs []types.Observation, node string) []types.Observation {
	var filtered []types.Observation
	for _, o := range obs {
		if o.Validator == node {
			filtered = append(filtered, o)
		}
	}
	return filtered
}

func getUniqueValidators(obs []types.Observation) []string {
	m := make(map[string]bool)
	var list []string
	for _, o := range obs {
		if !m[o.Validator] {
			m[o.Validator] = true
			list = append(list, o.Validator)
		}
	}
	return list
}

func getIntensities(obs []types.Observation) []float64 {
	var list []float64
	for _, o := range obs {
		f, _ := o.Intensity.Float64()
		list = append(list, f)
	}
	return list
}

func calculateStats(values []float64) (mean, std float64) {
	if len(values) == 0 {
		return 0, 0
	}
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	mean = sum / float64(len(values))

	sqDiffSum := 0.0
	for _, v := range values {
		sqDiffSum += math.Pow(v-mean, 2)
	}
	std = math.Sqrt(sqDiffSum / float64(len(values)))
	return mean, std
}

func extractCommonFeatures(obs []types.Observation) []string {
	if len(obs) == 0 {
		return nil
	}
	// Simplificação: quebra por espaço e busca interseção
	features := strings.Fields(obs[0].Context)
	intersect := features
	for i := 1; i < len(obs); i++ {
		intersect = intersection(intersect, strings.Fields(obs[i].Context))
	}
	return intersect
}

func intersection(a, b []string) []string {
	m := make(map[string]bool)
	for _, item := range a {
		m[item] = true
	}
	var res []string
	for _, item := range b {
		if m[item] {
			res = append(res, item)
		}
	}
	return res
}

func containsAll(context string, features []string) bool {
	ctxFeatures := strings.Fields(context)
	m := make(map[string]bool)
	for _, item := range ctxFeatures {
		m[item] = true
	}
	for _, f := range features {
		if !m[f] {
			return false
		}
	}
	return true
}

// Stubs

func (k Keeper) GetObservationsByPhenomenon(ctx sdk.Context, phenomenon string) []types.Observation {
	return nil
}

func (k Keeper) SubmitInductionProposal(ctx sdk.Context, instance types.PrerogativeInstance) {
	// Em produção, criaria uma proposta de governança
}
