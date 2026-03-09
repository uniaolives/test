package txpool

// peertick_compat.go re-exports types from txpool/peertick for backward compatibility.

import "arkhend/arkhen/eth2030/pkg/txpool/peertick"

// PeerTickCache type alias.
type PeerTickCache = peertick.PeerTickCache

// PeerTickCache function wrapper.
func NewPeerTickCache(slotTTL uint64) *PeerTickCache { return peertick.NewPeerTickCache(slotTTL) }
