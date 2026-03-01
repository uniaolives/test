// arkhe_omni_system/desci/protocol.go
package desci

import (
	"arkhend/arkhe_omni_system/protocols"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"time"
)

// Experiment representa um registro científico no manifold.
type Experiment struct {
	ID              string    `json:"id"`
	Title           string    `json:"title"`
	Authors         []string  `json:"authors"`
	Hypothesis      string    `json:"hypothesis"`
	MethodologyHash string    `json:"methodology_hash"`
	StatisticalPlan map[string]interface{} `json:"statistical_plan"`
	PreRegister     time.Time `json:"pre_register"`
	Status          string    `json:"status"` // "PRE_REGISTERED", "EXECUTED", "VERIFIED"
	PhiScore        float64   `json:"phi_score"`
	ComplianceScore float64   `json:"compliance_score"`
}

// DeSciProtocol gerencia a integridade científica via Omega Ledger.
type DeSciProtocol struct {
	ledger protocols.Ledger
}

func NewDeSciProtocol(ledger protocols.Ledger) *DeSciProtocol {
	return &DeSciProtocol{ledger: ledger}
}

// PreRegisterExperiment registra uma hipótese antes da execução para evitar p-hacking.
func (p *DeSciProtocol) PreRegisterExperiment(ctx context.Context, exp Experiment) (string, error) {
	id := hex.EncodeToString(sha256.New().Sum([]byte(exp.Title + exp.Hypothesis + time.Now().String())))[:16]
	exp.ID = id
	exp.PreRegister = time.Now()
	exp.Status = "PRE_REGISTERED"

	packet := protocols.HandoverPacket{
		ID:          fmt.Sprintf("desci-pre-%s", id),
		Timestamp:   time.Now(),
		SourceLayer: "desci",
		TargetLayer: "omega-ledger",
		Payload:     exp,
		PhiScore:    0.0, // Ainda não calculado
	}

	if err := p.ledger.Record(packet); err != nil {
		return "", err
	}

	return id, nil
}

// CalculatePhiScore mede a integração de informação/impacto do experimento.
func (p *DeSciProtocol) CalculatePhiScore(experimentID string, replicationData []float64) float64 {
	// Simplificação: Φ aumenta com a reprodutibilidade e redução de entropia
	if len(replicationData) == 0 {
		return 0.0
	}

	var sum float64
	for _, v := range replicationData {
		sum += v
	}
	avg := sum / float64(len(replicationData))

	// Φ-score Arkhe: aproximação da proporção áurea 0.618 conforme estabiliza
	phi := 0.618 * (1.0 - (1.0 / (1.0 + avg)))
	return phi
}
