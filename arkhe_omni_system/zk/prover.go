// arkhe_omni_system/zk/prover.go
package zk

import (
	"math/big"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark-crypto/hash"
	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
	"github.com/consensys/gnark/constraint"
)

// ZKCRDTProver gera provas ZK para transições CRDT.
type ZKCRDTProver struct {
	pk   groth16.ProvingKey
	vk   groth16.VerifyingKey
	r1cs constraint.ConstraintSystem
}

// NewZKCRDTProver inicializa o sistema ZK (setup de exemplo).
func NewZKCRDTProver() (*ZKCRDTProver, error) {
	var circuit GCounterTransitionCircuit
	ccs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, &circuit)
	if err != nil {
		return nil, err
	}

	pk, vk, err := groth16.Setup(ccs)
	if err != nil {
		return nil, err
	}

	return &ZKCRDTProver{
		pk:   pk,
		vk:   vk,
		r1cs: ccs,
	}, nil
}

// ProveTransition gera uma prova de que um incremento CRDT é válido.
func (zp *ZKCRDTProver) ProveTransition(
	oldState, increment uint64,
	salt *big.Int,
	oldHash, newHash, incHash *big.Int,
) (groth16.Proof, error) {

	assignment := GCounterTransitionCircuit{
		OldState:      oldState,
		Increment:     increment,
		Salt:          salt,
		OldStateHash:  oldHash,
		NewStateHash:  newHash,
		IncrementHash: incHash,
	}

	witness, err := frontend.NewWitness(&assignment, ecc.BN254.ScalarField())
	if err != nil {
		return nil, err
	}

	proof, err := groth16.Prove(zp.r1cs, zp.pk, witness)
	return proof, err
}

// Verify verifica uma prova ZK.
func (zp *ZKCRDTProver) Verify(
	proof groth16.Proof,
	oldHash, newHash, incHash *big.Int,
) error {
	publicAssignment := GCounterTransitionCircuit{
		OldStateHash:  oldHash,
		NewStateHash:  newHash,
		IncrementHash: incHash,
	}

	witness, err := frontend.NewWitness(&publicAssignment, ecc.BN254.ScalarField(), frontend.PublicOnly())
	if err != nil {
		return err
	}

	return groth16.Verify(proof, zp.vk, witness)
}

// ComputeMiMC é um utilitário para calcular hashes MiMC fora do circuito.
func ComputeMiMC(data ...*big.Int) *big.Int {
	f := hash.MIMC_BN254.New()
	for _, d := range data {
		f.Write(d.Bytes())
	}
	return new(big.Int).SetBytes(f.Sum(nil))
}

func Uint64ToBigInt(v uint64) *big.Int {
	return new(big.Int).SetUint64(v)
}
