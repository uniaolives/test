package rpc

// registry_compat.go re-exports types from rpc/registry for backward compatibility.

import "arkhend/arkhen/eth2030/pkg/rpc/registry"

// Registry type aliases.
type (
	MethodHandler  = registry.MethodHandler
	Middleware     = registry.Middleware
	MethodInfo     = registry.MethodInfo
	MethodRegistry = registry.MethodRegistry
)

// Registry error variables.
var (
	ErrMethodNotFound  = registry.ErrMethodNotFound
	ErrDuplicateMethod = registry.ErrDuplicateMethod
)

// Registry function wrappers.
func NewMethodRegistry() *MethodRegistry       { return registry.NewMethodRegistry() }
func NamespaceFromMethod(method string) string { return registry.NamespaceFromMethod(method) }
