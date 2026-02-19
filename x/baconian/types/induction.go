package types

import (
	"time"

	"cosmossdk.io/math"
)

type Dec = math.LegacyDec

type Status int

const (
	StatusDraft Status = iota
	StatusVoting
	StatusAccepted
	StatusRejected
)

type IdolType int

const (
	IdolumTribus IdolType = iota   // erros sensoriais sistemáticos
	IdolumSpecus                    // preconceitos individuais
	IdolumFori                      // distorções linguísticas
	IdolumTheatri                   // dogmas cristalizados
)

type Correction struct {
	Strategy   string
	Parameters []string
}

// Observation = handover sensorial registrado
type Observation struct {
	ID                  string
	Phenomenon          string      // fenômeno investigado
	Context             string      // condições do experimento
	Result              bool        // fenômeno ocorreu?
	Intensity           Dec         // grau de ocorrência (0-1)
	Timestamp           time.Time
	Hash                string      // hash do experimento (prova)
	Validator           string      // nó que realizou a observação
	Signature           []byte      // prova criptográfica
	CoherenceAtCreation Dec
}

type Table struct {
	Phenomenon   string
	Presence     []Observation  // instâncias positivas
	Absence      []Observation  // instâncias negativas
	Degrees      []Observation  // intensidades variadas
	ValidatorSet []string       // observadores únicos
}

// Idol = nó de baixa coerência identificado
type Idol struct {
	ID              string
	Type            IdolType    // Tribus, Specus, Fori, Theatri
	Severity        Dec         // [0,1]
	Description     string
	AffectedObs     []string    // IDs das observações afetadas
	AffectedNodes   []string    // nós que carregam este ídolo
	Correction      Correction  // método de correção
	CoherenceImpact Dec         // impacto negativo em C_global
}

type Vote struct {
	Validator     string
	Decision      bool
	Justification string
}

// PrerogativeInstance = handover de alta intensidade epistemológica
type PrerogativeInstance struct {
	ID          string
	Phenomenon  string           // fenômeno investigado
	Table       Table            // organização em tábuas
	Idols       []Idol           // ídolos detectados
	ProposedLaw string           // lei causal inferida
	Confidence  Dec              // confiança na inferência (0-1)
	Support     Dec              // cobertura
	Specificity Dec              // precisão
	Status      Status           // Draft|Voting|Accepted|Rejected
	Votes       []Vote           // votos dos validadores
	Validator   string           // nó que propôs
	BlockHeight int64
}
