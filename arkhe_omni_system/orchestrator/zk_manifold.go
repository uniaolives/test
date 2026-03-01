// arkhe_omni_system/orchestrator/zk_manifold.go
package orchestrator

import (
	"arkhend/arkhe_omni_system/crdt"
	"arkhend/arkhe_omni_system/zk"
	"arkhend/arkhe_omni_system/protocols"
	"context"
	"crypto/rand"
	"fmt"
	"math/big"
	"time"

	"github.com/consensys/gnark/backend/groth16"
)

// OmegaLedger interface baseada no que vimos no OMEGA_LEDGER
type OmegaLedger interface {
	Record(ctx context.Context, packet protocols.HandoverPacket) error
}

// ZKStateShard encapsula um CRDT com sua camada ZK.
type ZKStateShard struct {
	ID      string
	Counter *crdt.GCounter
	Prover  *zk.ZKCRDTProver

	// Estado privado (hashes e salts)
	CurrentSalt *big.Int
}

// ZKArkheMultiCloudManifold gerencia a orquestração ZK-CRDT.
type ZKArkheMultiCloudManifold struct {
	nodeID crdt.NodeID
	shards map[string]*ZKStateShard
	ledger OmegaLedger
}

func NewZKArkheMultiCloudManifold(nodeID crdt.NodeID, ledger OmegaLedger) *ZKArkheMultiCloudManifold {
	return &ZKArkheMultiCloudManifold{
		nodeID: nodeID,
		shards: make(map[string]*ZKStateShard),
		ledger: ledger,
	}
}

// AddShard inicializa um novo shard de estado.
func (m *ZKArkheMultiCloudManifold) AddShard(id string) error {
	prover, err := zk.NewZKCRDTProver()
	if err != nil {
		return err
	}

	m.shards[id] = &ZKStateShard{
		ID:      id,
		Counter: crdt.NewGCounter(m.nodeID),
		Prover:  prover,
	}
	return nil
}

// ApplyPrivateTransition incrementa o estado e gera uma prova ZK.
func (m *ZKArkheMultiCloudManifold) ApplyPrivateTransition(
	ctx context.Context,
	shardID string,
	increment uint64,
) (groth16.Proof, error) {
	shard, exists := m.shards[shardID]
	if !exists {
		return nil, fmt.Errorf("shard %s não encontrado", shardID)
	}

	oldValue := shard.Counter.Value()

	// Gera novo salt aleatório
	salt, _ := rand.Int(rand.Reader, big.NewInt(1).Lsh(big.NewInt(1), 253)) // MiMC safe range
	shard.CurrentSalt = salt

	// Calcula hashes para inputs públicos
	oldValBI := zk.Uint64ToBigInt(oldValue)
	incBI := zk.Uint64ToBigInt(increment)
	newValBI := zk.Uint64ToBigInt(oldValue + increment)

	oldHash := zk.ComputeMiMC(oldValBI, salt)
	newHash := zk.ComputeMiMC(newValBI, salt)
	incHash := zk.ComputeMiMC(incBI, salt)

	// Gera prova
	proof, err := shard.Prover.ProveTransition(oldValue, increment, salt, oldHash, newHash, incHash)
	if err != nil {
		return nil, err
	}

	// Aplica localmente
	shard.Counter.Increment(increment)

	// Registra no Ledger (Público)
	packet := protocols.HandoverPacket{
		ID:          fmt.Sprintf("zk-%s-%d", shardID, time.Now().UnixNano()),
		Timestamp:   time.Now(),
		SourceLayer: "zk-manifold",
		TargetLayer: "omega-ledger",
		Payload: map[string]interface{}{
			"shard":     shardID,
			"old_hash":  oldHash.String(),
			"new_hash":  newHash.String(),
			"inc_hash":  incHash.String(),
			"proof":     "groth16-encoded-data", // Em real, serializaria a prova
		},
		PhiScore: 0.618, // Ideal Arkhe
	}

	m.ledger.Record(ctx, packet)

	return proof, nil
}
