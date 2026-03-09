package state

// acctcache_compat.go re-exports types from core/state/acctcache for backward compatibility.

import "arkhend/arkhen/eth2030/pkg/core/state/acctcache"

// AccountCache type aliases.
type (
	AccountCache = acctcache.AccountCache
)

// AccountCache function wrappers.
func NewAccountCache(maxSize int) *AccountCache { return acctcache.NewAccountCache(maxSize) }
