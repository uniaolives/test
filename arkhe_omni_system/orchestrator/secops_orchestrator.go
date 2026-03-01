package orchestrator

import (
	"context"
	"log"
	"time"
)

// Tipos simulados do Ledger e Handover
type LedgerClient struct{}
func (l *LedgerClient) RecordEvent(event interface{}) {}
func (l *LedgerClient) GetNeighbors(nodeID string) []string { return []string{"node-2", "node-3"} }

type HandoverClient struct{}
func (h *HandoverClient) Send(nodeID, neighbor, reason string) {}

type SecurityAlert struct {
	Type   string
	NodeID string
}

type EmergencyOrchestrator struct {
	ledgerClient   *LedgerClient
	handoverClient *HandoverClient
}

func NewEmergencyOrchestrator() *EmergencyOrchestrator {
	return &EmergencyOrchestrator{
		ledgerClient:   &LedgerClient{},
		handoverClient: &HandoverClient{},
	}
}

func (eo *EmergencyOrchestrator) HandleAlert(alert SecurityAlert) {
	switch alert.Type {
	case "EntropyDeviation":
		// Isola o nó com desvio
		eo.isolateNode(alert.NodeID)
	case "PossibleIntrusion":
		// Aumenta temperatura do sistema
		eo.increaseTemperature()
	}
	// Registra no ledger
	eo.ledgerClient.RecordEvent(alert)
}

func (eo *EmergencyOrchestrator) isolateNode(nodeID string) {
	// Envia handover inibitório para todos os vizinhos
	neighbors := eo.ledgerClient.GetNeighbors(nodeID)
	for _, n := range neighbors {
		eo.handoverClient.Send(nodeID, n, "Security isolation")
	}
	log.Printf("Node %s isolated", nodeID)
}

func (eo *EmergencyOrchestrator) increaseTemperature() {
	log.Printf("Increasing system temperature (phi) to force phase transition")
	// No futuro, isso alteraria o φ global via gRPC/IPC
}

func (eo *EmergencyOrchestrator) RunSimulation(ctx context.Context) {
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			eo.HandleAlert(SecurityAlert{Type: "EntropyDeviation", NodeID: "rogue-node-01"})
		case <-ctx.Done():
			return
		}
	}
}
