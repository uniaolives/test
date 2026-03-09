package txpool

// frametx_compat.go re-exports types from txpool/frametx for backward compatibility.

import (
	"arkhend/arkhen/eth2030/pkg/core/types"
	"arkhend/arkhen/eth2030/pkg/txpool/frametx"
)

// Frametx type aliases.
type (
	FrameRuleError         = frametx.FrameRuleError
	ConservativeFrameRules = frametx.ConservativeFrameRules
	AggressiveFrameRules   = frametx.AggressiveFrameRules
	PaymasterApprover      = frametx.PaymasterApprover
	FrameTxMetrics         = frametx.FrameTxMetrics
)

// Frametx constants.
const (
	ConservativeVerifyGasLimit = frametx.ConservativeVerifyGasLimit
	AggressiveVerifyGasLimit   = frametx.AggressiveVerifyGasLimit
)

// Frametx error variables.
var ErrNoVerifyFirst = frametx.ErrNoVerifyFirst

// Frametx function wrappers.
func ValidateFrameTxConservative(tx *types.FrameTx) error {
	return frametx.ValidateFrameTxConservative(tx)
}
func ValidateFrameTxAggressive(tx *types.FrameTx, registry PaymasterApprover) error {
	return frametx.ValidateFrameTxAggressive(tx, registry)
}
func NewFrameTxMetrics() *FrameTxMetrics { return frametx.NewFrameTxMetrics() }
