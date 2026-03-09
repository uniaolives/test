package consensus

// types_compat.go re-exports types from consensus/cltypes for backward compatibility.

import "arkhend/arkhen/eth2030/pkg/consensus/cltypes"

// Type aliases.
type (
	Epoch             = cltypes.Epoch
	Slot              = cltypes.Slot
	ValidatorIndex    = cltypes.ValidatorIndex
	Checkpoint        = cltypes.Checkpoint
	JustificationBits = cltypes.JustificationBits
	BeaconState       = cltypes.BeaconState
)

// Function wrappers.
func SlotToEpoch(slot Slot, slotsPerEpoch uint64) Epoch {
	return cltypes.SlotToEpoch(slot, slotsPerEpoch)
}

func EpochStartSlot(epoch Epoch, slotsPerEpoch uint64) Slot {
	return cltypes.EpochStartSlot(epoch, slotsPerEpoch)
}
