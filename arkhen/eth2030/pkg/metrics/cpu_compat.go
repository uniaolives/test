package metrics

// cpu_compat.go re-exports types from metrics/cpu for backward compatibility.

import "arkhend/arkhen/eth2030/pkg/metrics/cpu"

// CPU type aliases.
type (
	CPUStats   = cpu.CPUStats
	CPUTracker = cpu.CPUTracker
)

// CPU function wrappers.
func ReadCPUStats() *CPUStats    { return cpu.ReadCPUStats() }
func NewCPUTracker() *CPUTracker { return cpu.NewCPUTracker() }
