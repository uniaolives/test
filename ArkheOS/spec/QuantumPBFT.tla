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

(* Safety and Liveness properties are refined from QuantumPaxos *)
=======================================================
