----------------- MODULE QuantumPaxos -----------------
(* Quantum Paxos with Authenticated Messages (N=4, f=1) *)
EXTENDS Integers, Sequences, FiniteSets

CONSTANTS Nodes, QuorumSize, Value

VARIABLES state_log, current_ballot, messages

TypeInvariant ==
    /\ state_log \in [Nat -> {NULL} \cup Value]
    /\ current_ballot \in [Nodes -> Nat]
    /\ messages \in SUBSET [type: STRING, sender: Nodes, ballot: Nat, slot: Nat, value: Value]

(* Safety Property *)
Consistency == \A s \in Nat, v1, v2 \in Value:
    (state_log[s] = v1 /\ state_log[s] = v2) => (v1 = v2)

(* Liveness Property with Leader Timeout *)
Liveness == \A s \in Nat:
    (state_log[s] = NULL) ~> (\E v \in Value: state_log[s] = v)

=======================================================
