package consensus

// config_compat.go re-exports types from consensus/clconfig for backward compatibility.

import "arkhend/arkhen/eth2030/pkg/consensus/clconfig"

// ConsensusConfig type alias.
type ConsensusConfig = clconfig.ConsensusConfig

// ConsensusConfig function wrappers.
func DefaultConfig() *ConsensusConfig    { return clconfig.DefaultConfig() }
func QuickSlotsConfig() *ConsensusConfig { return clconfig.QuickSlotsConfig() }
