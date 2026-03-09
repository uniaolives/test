package engine

// errors_compat.go re-exports errors and codes from engine/apierrors for backward compatibility.

import "arkhend/arkhen/eth2030/pkg/engine/apierrors"

// Engine API errors.
var (
	ErrInvalidParams            = apierrors.ErrInvalidParams
	ErrUnknownPayload           = apierrors.ErrUnknownPayload
	ErrInvalidForkchoiceState   = apierrors.ErrInvalidForkchoiceState
	ErrInvalidPayloadAttributes = apierrors.ErrInvalidPayloadAttributes
	ErrTooLargeRequest          = apierrors.ErrTooLargeRequest
	ErrUnsupportedFork          = apierrors.ErrUnsupportedFork
	ErrInvalidBlockHash         = apierrors.ErrInvalidBlockHash
	ErrInvalidBlobHashes        = apierrors.ErrInvalidBlobHashes
	ErrMissingBeaconRoot        = apierrors.ErrMissingBeaconRoot
)

// Standard JSON-RPC 2.0 error codes.
const (
	ParseErrorCode     = apierrors.ParseErrorCode
	InvalidRequestCode = apierrors.InvalidRequestCode
	MethodNotFoundCode = apierrors.MethodNotFoundCode
	InvalidParamsCode  = apierrors.InvalidParamsCode
	InternalErrorCode  = apierrors.InternalErrorCode
)

// Engine API specific error codes.
const (
	UnknownPayloadCode          = apierrors.UnknownPayloadCode
	InvalidForkchoiceStateCode  = apierrors.InvalidForkchoiceStateCode
	InvalidPayloadAttributeCode = apierrors.InvalidPayloadAttributeCode
	TooLargeRequestCode         = apierrors.TooLargeRequestCode
	UnsupportedForkCode         = apierrors.UnsupportedForkCode
)
