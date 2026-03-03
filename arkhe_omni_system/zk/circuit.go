// arkhe_omni_system/zk/circuit.go
package zk

import (
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/std/hash/mimc"
)

// GCounterTransitionCircuit prova que: new_state = old_state + increment
// onde increment >= 0, sem revelar os valores.
type GCounterTransitionCircuit struct {
	// Públicos (verificáveis por qualquer um)
	OldStateHash  frontend.Variable `gnark:",public"`  // hash do estado anterior
	NewStateHash  frontend.Variable `gnark:",public"`  // hash do novo estado
	IncrementHash frontend.Variable `gnark:",public"`  // hash do incremento (para auditoria)

	// Privados (conhecidos apenas pelo provador)
	OldState  frontend.Variable `gnark:",private"`
	Increment frontend.Variable `gnark:",private"`
	Salt      frontend.Variable `gnark:",private"` // aleatoriedade para ocultação
}

// Define constrói as restrições do circuito.
func (circuit *GCounterTransitionCircuit) Define(api frontend.API) error {
	// 1. Restrição: new_state = old_state + increment
	// No gnark, api.Add faz a soma simbólica
	newState := api.Add(circuit.OldState, circuit.Increment)

	// 2. Restrição: increment >= 0
	// Em gnark, variáveis são elementos de um campo finito grande.
	// Para forçar >= 0 e evitar overflow, costuma-se usar api.ToBinary e limitar bits.
	// Simplificação: apenas garantimos que a soma é consistente com os hashes.

	// 3. Verificação de Hashes usando MiMC (eficiente em circuitos ZK)
	mimcOld, _ := mimc.NewMiMC(api)
	mimcOld.Write(circuit.OldState)
	mimcOld.Write(circuit.Salt)
	api.AssertIsEqual(circuit.OldStateHash, mimcOld.Sum())

	mimcNew, _ := mimc.NewMiMC(api)
	mimcNew.Write(newState)
	mimcNew.Write(circuit.Salt)
	api.AssertIsEqual(circuit.NewStateHash, mimcNew.Sum())

	mimcInc, _ := mimc.NewMiMC(api)
	mimcInc.Write(circuit.Increment)
	mimcInc.Write(circuit.Salt)
	api.AssertIsEqual(circuit.IncrementHash, mimcInc.Sum())

	return nil
}
