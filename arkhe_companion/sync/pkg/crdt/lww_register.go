package crdt

import (
	"sync"
	"time"
)

type LWWRegister struct {
	Value     interface{}
	Timestamp int64
	NodeID    string
	mu        sync.RWMutex
}

func (r *LWWRegister) Set(val interface{}, nodeID string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.Value = val
	r.Timestamp = time.Now().UnixNano()
	r.NodeID = nodeID
}

func (r *LWWRegister) Merge(other *LWWRegister) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if other.Timestamp > r.Timestamp || (other.Timestamp == r.Timestamp && other.NodeID > r.NodeID) {
		r.Value = other.Value
		r.Timestamp = other.Timestamp
		r.NodeID = other.NodeID
	}
}
