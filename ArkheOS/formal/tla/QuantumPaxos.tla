------------------------------ MODULE QuantumPaxos ------------------------------
EXTENDS Naturals, FiniteSets, Sequences

CONSTANTS
    Nodes,          \* Conjunto de nós {q1, q2, q3}
    QuorumSize,     \* 2f+1, para N=3, f=0 → 2; f=1 → 2 (tolerância bizantina)
    Value           \* Tipo abstrato para estados quânticos

VARIABLES
    ballot,         \* [Node → Nat] – último ballot preparado
    slot,           \* [Node → Nat] – próximo slot disponível
    promises,       \* [Node → Set of (Node, Nat)] – promessas recebidas
    accepts,        \* [Node → Set of (Node, Nat, Value)] – accepts recebidos
    decision,       \* [Node → Seq(Value)] – histórico de decisões
    state_log       \* [Slot → Value] – estado global commitado

vars == <<ballot, slot, promises, accepts, decision, state_log>>

\* ---------- Tipos e Invariantes ----------
TypeInvariant ==
    /\ ballot ∈ [Nodes → Nat]
    /\ slot ∈ [Nodes → Nat]
    /\ ∀ n ∈ Nodes: promises[n] ⊆ {m ∈ Nodes: m ≠ n} × Nat
    /\ ∀ n ∈ Nodes: accepts[n] ⊆ {m ∈ Nodes: m ≠ n} × Nat × Value
    /\ decision ∈ [Nodes → Seq(Value)]
    /\ state_log ∈ [Nat → Value]

\* ---------- Ações ----------
Propose(n) ==
    /\ ballot' = [ballot EXCEPT ![n] = ballot[n] + 1]
    /\ UNCHANGED <<slot, promises, accepts, decision, state_log>>

Promise(n, m, b) ==
    /\ b = ballot[m]
    /\ n ≠ m
    /\ promises' = [promises EXCEPT ![n] = promises[n] ∪ {<<m, b>>}]
    /\ UNCHANGED <<ballot, slot, accepts, decision, state_log>>

Accept(n, m, b, v) ==
    /\ b = ballot[m]
    /\ n ≠ m
    /\ accepts' = [accepts EXCEPT ![n] = accepts[n] ∪ {<<m, b, v>>}]
    /\ UNCHANGED <<ballot, slot, promises, decision, state_log>>

Learn(n, s, v) ==
    /\ decision' = [decision EXCEPT ![n] = Append(decision[n], <<s, v>>)]
    /\ UNCHANGED <<ballot, slot, promises, accepts, state_log>>

\* ---------- Especificação ----------
Init ==
    /\ ballot = [n ∈ Nodes ↦ 0]
    /\ slot = [n ∈ Nodes ↦ 0]
    /\ promises = [n ∈ Nodes ↦ {}]
    /\ accepts = [n ∈ Nodes ↦ {}]
    /\ decision = [n ∈ Nodes ↦ << >>]
    /\ state_log = [i ∈ Nat ↦ CHOOSE v: v = v]  \* qualquer valor inicial

Next ==
    \/ ∃ n ∈ Nodes: Propose(n)
    \/ ∃ n, m ∈ Nodes, b ∈ Nat: Promise(n, m, b)
    \/ ∃ n, m ∈ Nodes, b ∈ Nat, v ∈ Value: Accept(n, m, b, v)
    \/ ∃ n ∈ Nodes, s ∈ Nat, v ∈ Value: Learn(n, s, v)

Spec == Init /\ [][Next]_vars

\* ---------- Propriedades ----------
Safety == \forall s \in Nat, v1, v2 \in Value:
    /\ state_log[s] = v1
    /\ state_log[s] = v2
    => v1 = v2

Liveness == \forall s \in Nat:
    <> \exists v \in Value: state_log[s] = v

===============================================================================
