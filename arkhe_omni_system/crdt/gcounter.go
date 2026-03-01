// arkhe_omni_system/crdt/gcounter.go
package crdt

import (
	"encoding/json"
	"sync"
)

// NodeID identifica unicamente um nó no cluster multi-cloud.
type NodeID string

// GCounter implementa um contador crescente livre de conflitos.
// Matemática: Semigrupo comutativo (N, +) elevado ao espaço de nós.
type GCounter struct {
	// id é o identificador deste nó (ex: "aws-us-east-1", "azure-westeurope")
	id NodeID

	// state é o vetor de contagens: state[node] = incrementos feitos por 'node'
	// Invariante: state[id] é a única posição que este nó pode incrementar.
	state map[NodeID]uint64

	mu sync.RWMutex // proteção para concorrência local
}

// NewGCounter inicializa um contador vazio para o nó especificado.
func NewGCounter(id NodeID) *GCounter {
	return &GCounter{
		id:    id,
		state: make(map[NodeID]uint64),
	}
}

// Increment adiciona 'delta' à contagem local deste nó.
// Operação local, nunca falha, nunca bloqueia outros nós.
func (c *GCounter) Increment(delta uint64) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.state[c.id] += delta
}

// Value retorna o valor total do contador (soma de todas as réplicas).
func (c *GCounter) Value() uint64 {
	c.mu.RLock()
	defer c.mu.RUnlock()

	var sum uint64
	for _, v := range c.state {
		sum += v
	}
	return sum
}

// Merge implementa a operação de "trança" (braiding) do CRDT.
// Recebe o estado de outro nó e combina de forma comutativa.
// Propriedade: merge(A, B) == merge(B, A) (comutatividade)
// Propriedade: merge(merge(A, B), C) == merge(A, merge(B, C)) (associatividade)
func (c *GCounter) Merge(other *GCounter) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Para cada nó conhecido pelo 'other', tomamos o máximo.
	// Isso garante que não perdemos incrementos, independente da ordem.
	for node, count := range other.state {
		if current, exists := c.state[node]; !exists || count > current {
			c.state[node] = count
		}
	}
}

// ToJSON serializa o estado para transmissão via rede.
func (c *GCounter) ToJSON() ([]byte, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return json.Marshal(c.state)
}

// FromJSON desserializa e funde um estado recebido da rede.
func (c *GCounter) FromJSON(data []byte) error {
	var received map[NodeID]uint64
	if err := json.Unmarshal(data, &received); err != nil {
		return err
	}

	// Criamos um GCounter temporário para fazer o merge
	c.mu.Lock()
	defer c.mu.Unlock()
	for node, count := range received {
		if current, exists := c.state[node]; !exists || count > current {
			c.state[node] = count
		}
	}
	return nil
}
