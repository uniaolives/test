----------------- MODULE QuantumPBFT -----------------
(* Practical Byzantine Fault Tolerance for Quantum Arkhe(n) *)
EXTENDS Integers, Sequences, FiniteSets

CONSTANTS Nodes, F, Value

VARIABLES round, prepareQC, lockedQC, node_status

(* N = 3F + 1 *)
ASSUME Cardinality(Nodes) = 3*F + 1

TypeInvariant ==
    /\ round \in Nat
    /\ node_status \in [Nodes -> {"honest", "byzantine"}]

(* Quorum Certificate (QC) requires 2F+1 signatures *)
IsQC(qc) == Cardinality(qc.sigs) >= 2*F + 1

(* Quorum Intersection Property: any two QCs of 2F+1 have at least one honest node in common *)
QuorumIntersection == \A q1, q2 \in SUBSET Nodes:
    (Cardinality(q1) >= 2*F + 1 /\ Cardinality(q2) >= 2*F + 1)
    => \E n \in q1 \cap q2: node_status[n] = "honest"

(* Safety and Liveness properties are refined from QuantumPaxos *)
=======================================================
