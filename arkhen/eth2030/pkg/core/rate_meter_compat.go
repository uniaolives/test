package core

// rate_meter_compat.go re-exports types from core/ratemeter for backward compatibility.

import "arkhend/arkhen/eth2030/pkg/core/ratemeter"

// Type aliases.
type (
	RateMeterConfig = ratemeter.RateMeterConfig
	RateMeter       = ratemeter.RateMeter
)

// Function wrappers.
func DefaultRateMeterConfig() RateMeterConfig { return ratemeter.DefaultRateMeterConfig() }
func NewRateMeter(config RateMeterConfig) *RateMeter {
	return ratemeter.NewRateMeter(config)
}
