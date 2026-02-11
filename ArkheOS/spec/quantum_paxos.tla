------------------- MODULE quantum_paxos -------------------
EXTENDS Integers, Sequences, FiniteSets

CONSTANT Nodes, TotalNodes, FaultyNodes, MaxBallot

ASSUME FaultyNodes * 3 < TotalNodes
ASSUME Nodes \subseteq DOMAIN [1..TotalNodes]

VARIABLES
    ballot,          \* current ballot number for each node
    slot,            \* current slot being decided
    state,           \* local state vector commit
    messages,        \* set of messages in transit
    promises,        \* promises received per node
    accepts,         \* accepts received per node
    log              \* committed state log

vars == <<ballot, slot, state, messages, promises, accepts, log>>

TypeOK ==
    /\ ballot \in [Nodes -> 0..MaxBallot]
    /\ slot \in [Nodes -> Int]
    /\ messages \in SUBSET [type: {"PREPARE", "PROMISE", "ACCEPT", "ACCEPTED"},
                            node: Nodes, ballot: 0..MaxBallot, slot: Int, value: SUBSET {"state_vec"}]

Init ==
    /\ ballot = [n \in Nodes -> 0]
    /\ slot = [n \in Nodes -> 0]
    /\ state = [n \in Nodes -> "neutral"]
    /\ messages = {}
    /\ promises = [n \in Nodes -> {}]
    /\ accepts = [n \in Nodes -> {}]
    /\ log = [n \in Nodes -> << >>]

Send(m) == messages' = messages \cup {m}

Prepare(n) ==
    /\ ballot[n] < MaxBallot
    /\ ballot' = [ballot EXCEPT ![n] = ballot[n] + 1]
    /\ Send([type |-> "PREPARE", node |-> n, ballot |-> ballot'[n], slot |-> slot[n]])
    /\ UNCHANGED <<slot, state, promises, accepts, log>>

Promise(n, m) ==
    /\ m.type = "PREPARE"
    /\ m.ballot > ballot[n]
    /\ ballot' = [ballot EXCEPT ![n] = m.ballot]
    /\ Send([type |-> "PROMISE", node |-> n, ballot |-> m.ballot, slot |-> m.slot])
    /\ UNCHANGED <<slot, state, promises, accepts, log>>

Accept(n) ==
    /\ Cardinality(promises[n]) >= 2 * FaultyNodes + 1
    /\ Send([type |-> "ACCEPT", node |-> n, ballot |-> ballot[n], slot |-> slot[n], value |-> "state_vec"])
    /\ UNCHANGED <<ballot, slot, state, promises, accepts, log>>

Accepted(n, m) ==
    /\ m.type = "ACCEPT"
    /\ m.ballot >= ballot[n]
    /\ Send([type |-> "ACCEPTED", node |-> n, ballot |-> m.ballot, slot |-> m.slot, value |-> m.value])
    /\ UNCHANGED <<ballot, slot, state, promises, accepts, log>>

Commit(n) ==
    /\ Cardinality(accepts[n]) >= 2 * FaultyNodes + 1
    /\ log' = [log EXCEPT ![n] = Append(log[n], [slot |-> slot[n], value |-> "state_vec"])]
    /\ slot' = [slot EXCEPT ![n] = slot[n] + 1]
    /\ UNCHANGED <<ballot, state, messages, promises, accepts>>

Next ==
    \exists n \in Nodes :
        \/ Prepare(n)
        \/ Commit(n)
        \/ \exists m \in messages :
            \/ Promise(n, m)
            \/ Accepted(n, m)

Agreement == \forall n1, n2 \in Nodes :
    \forall i \in 1..Min(Len(log[n1]), Len(log[n2])) :
        log[n1][i] = log[n2][i]

Min(a, b) == if a < b then a else b

=============================================================
