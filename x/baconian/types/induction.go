package types

import (
	"time"

	"cosmossdk.io/math"
)

type Dec = math.LegacyDec

// PrerogativeInstance = handover de alta intensidade epistemológica
type PrerogativeInstance struct {
	ID          string
	Phenomenon  string           // fenômeno investigado
	Presence    []Observation    // tábua de presença
	Absence     []Observation    // tábua de ausência
	Degrees     []Observation    // tábua de graus
	ProposedLaw string           // lei causal inferida
	Confidence  Dec              // confiança na inferência (0-1)
	Validator   string           // nó que propôs
	BlockHeight int64
}

// Observation = handover sensorial registrado
type Observation struct {
	Context    string      // condições do experimento
	Result     bool        // fenômeno ocorreu?
	Intensity  Dec         // grau de ocorrência (0-1)
	Timestamp  time.Time
	Hash       string      // hash do experimento (prova)
	Validator  string      // nó que realizou a observação
}

// Idol = nó de baixa coerência identificado
type Idol struct {
	Type          IdolType    // Tribus, Specus, Fori, Theatri
	Description   string
	AffectedNodes []string  // nós que carregam este ídolo
	Correction    string      // método de correção
}

type IdolType int

const (
	IdolumTribus IdolType = iota   // erros sensoriais sistemáticos
	IdolumSpecus                    // preconceitos individuais
	IdolumFori                      // distorções linguísticas
	IdolumTheatri                   // dogmas cristalizados
)
