package core

// gaspool_compat.go re-exports types from core/gaspool for backward compatibility.

import "arkhend/arkhen/eth2030/pkg/core/gaspool"

// GasPool type alias.
type GasPool = gaspool.GasPool

// GasPool error variable.
var ErrGasPoolExhausted = gaspool.ErrGasPoolExhausted
