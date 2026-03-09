package trie

// nodecache_compat.go re-exports types from trie/nodecache for backward compatibility.

import "arkhend/arkhen/eth2030/pkg/trie/nodecache"

// Cache type aliases.
type (
	CacheStats = nodecache.CacheStats
	TrieCache  = nodecache.TrieCache
)

// Cache function wrappers.
func NewTrieCache(maxSize int) *TrieCache { return nodecache.NewTrieCache(maxSize) }
