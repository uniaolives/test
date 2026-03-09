package rlp

// errors_compat.go re-exports errors from rlp/rlperrors for backward compatibility.

import "arkhend/arkhen/eth2030/pkg/rlp/rlperrors"

var (
	ErrExpectedString   = rlperrors.ErrExpectedString
	ErrExpectedList     = rlperrors.ErrExpectedList
	ErrCanonSize        = rlperrors.ErrCanonSize
	ErrEOL              = rlperrors.ErrEOL
	ErrCanonInt         = rlperrors.ErrCanonInt
	ErrNonCanonicalSize = rlperrors.ErrNonCanonicalSize
	ErrUint64Range      = rlperrors.ErrUint64Range
	ErrValueTooLarge    = rlperrors.ErrValueTooLarge
)
