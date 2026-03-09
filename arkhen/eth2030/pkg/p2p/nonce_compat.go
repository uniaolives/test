package p2p

// nonce_compat.go re-exports types from p2p/nonce for backward compatibility.

import (
	"time"

	"arkhend/arkhen/eth2030/pkg/p2p/nonce"
)

// Nonce type aliases.
type (
	NonceRecord    = nonce.NonceRecord
	NonceCache     = nonce.NonceCache
	NonceAnnouncer = nonce.NonceAnnouncer
)

// Nonce constants.
const (
	DefaultNonceCacheSize = nonce.DefaultNonceCacheSize
	DefaultNonceTTL       = nonce.DefaultNonceTTL
	DefaultMaxPeers       = nonce.DefaultMaxPeers
)

// Nonce error variables.
var (
	ErrNonceEmpty     = nonce.ErrNonceEmpty
	ErrNonceZeroHash  = nonce.ErrNonceZeroHash
	ErrNonceDuplicate = nonce.ErrNonceDuplicate
	ErrNonceTooMany   = nonce.ErrNonceTooMany
	ErrNonceNotFound  = nonce.ErrNonceNotFound
)

// Nonce function wrappers.
func NewNonceCache(maxSize int, ttl time.Duration) *NonceCache {
	return nonce.NewNonceCache(maxSize, ttl)
}
func NewNonceAnnouncer() *NonceAnnouncer { return nonce.NewNonceAnnouncer() }
func NewNonceAnnouncerWithConfig(cacheSize int, ttl time.Duration, maxPeers int) *NonceAnnouncer {
	return nonce.NewNonceAnnouncerWithConfig(cacheSize, ttl, maxPeers)
}
