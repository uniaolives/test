package bal

import (
	"arkhend/arkhen/eth2030/pkg/core/types"
	"arkhend/arkhen/eth2030/pkg/crypto"
	"arkhend/arkhen/eth2030/pkg/rlp"
)

// EncodeRLP returns the RLP encoding of the BlockAccessList.
func (bal *BlockAccessList) EncodeRLP() ([]byte, error) {
	return rlp.EncodeToBytes(bal)
}

// Hash computes the Keccak256 hash of the RLP-encoded BlockAccessList.
func (bal *BlockAccessList) Hash() types.Hash {
	encoded, err := bal.EncodeRLP()
	if err != nil {
		return types.Hash{}
	}
	return crypto.Keccak256Hash(encoded)
}
