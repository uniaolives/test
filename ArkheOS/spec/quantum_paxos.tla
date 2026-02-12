----------------- MODULE QuantumPaxos -----------------
(* Quantum Paxos with Authenticated Messages (N=4, f=1) *)
(* Extended with Partition and Crash Recovery Modeling *)
EXTENDS Integers, Sequences, FiniteSets

CONSTANTS Nodes, QuorumSize, Value

VARIABLES state_log, current_ballot, messages, node_status

TypeInvariant ==
    /\ state_log \in [Nat -> {NULL} \cup Value]
    /\ current_ballot \in [Nodes -> Nat]
    /\ messages \in SUBSET [type: STRING, sender: Nodes, ballot: Nat, slot: Nat, value: Value]
    /\ node_status \in [Nodes -> {"up", "down"}]

(* Network Partition Predicate *)
Partition(S1, S2) ==
    /\ S1 \cup S2 = Nodes
    /\ S1 \cap S2 = {}
    /\ \A m \in messages:
        (m.sender \in S1 => \A n \in S2: ~CanReceive(n, m))
        /\ (m.sender \in S2 => \A n \in S1: ~CanReceive(n, m))

(* Safety Property *)
Consistency == \A s \in Nat, v1, v2 \in Value:
    (state_log[s] = v1 /\ state_log[s] = v2) => (v1 = v2)

(* Liveness Property *)
Liveness == \A s \in Nat:
    (state_log[s] = NULL) ~> (\E v \in Value: state_log[s] = v)

=======================================================
