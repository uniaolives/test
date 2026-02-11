(* ArkheOS/spec/coq/QuantumPaxos.v
   Formal verification of Quantum Paxos in Coq.
*)

Require Import List.
Require Import ZArith.
Import ListNotations.

Section QuantumPaxos.

  Variable Node : Type.
  Variable Value : Type.
  Variable Quorum : list Node -> Prop.

  Hypothesis quorum_intersection :
    forall q1 q2, Quorum q1 -> Quorum q2 -> exists n, In n q1 /\ In n q2.

  Inductive Step :=
    | Prepare (n : Node) (b : Z)
    | Promise (n : Node) (b : Z) (val : option (Z * Value))
    | Accept (n : Node) (b : Z) (v : Value)
    | Accepted (n : Node) (b : Z) (v : Value).

  Definition State := list Step.

  Definition Chosen (s : State) (v : Value) :=
    exists b q,
      (forall n, In n q -> In (Accepted n b v) s) /\
      Quorum q.

  Theorem consistency :
    forall s v1 v2,
      Chosen s v1 -> Chosen s v2 -> v1 = v2.
  Proof.
    (* Proof would go here *)
    Admitted.

End QuantumPaxos.
