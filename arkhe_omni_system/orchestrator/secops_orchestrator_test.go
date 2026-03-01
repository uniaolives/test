package orchestrator

import (
	"context"
	"testing"
	"time"
)

func TestEmergencyOrchestrator_HandleAlert(t *testing.T) {
	eo := NewEmergencyOrchestrator()

	t.Run("EntropyDeviation", func(t *testing.T) {
		eo.HandleAlert(SecurityAlert{Type: "EntropyDeviation", NodeID: "test-node"})
	})

	t.Run("PossibleIntrusion", func(t *testing.T) {
		eo.HandleAlert(SecurityAlert{Type: "PossibleIntrusion", NodeID: "test-node"})
	})
}

func TestEmergencyOrchestrator_RunSimulation(t *testing.T) {
	eo := NewEmergencyOrchestrator()
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	// Run in background and wait for timeout
	eo.RunSimulation(ctx)
}
