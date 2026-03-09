package zkvm

// types_compat.go re-exports types from zkvm/zkvmtypes for backward compatibility.

import "arkhend/arkhen/eth2030/pkg/zkvm/zkvmtypes"

// zkVM type aliases.
type (
	GuestProgram    = zkvmtypes.GuestProgram
	VerificationKey = zkvmtypes.VerificationKey
	Proof           = zkvmtypes.Proof
	ProverBackend   = zkvmtypes.ProverBackend
	ExecutionResult = zkvmtypes.ExecutionResult
	GuestInput      = zkvmtypes.GuestInput
)
