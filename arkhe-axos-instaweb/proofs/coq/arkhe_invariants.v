(* arkhe_invariants.v - Provas formais dos invariantes Arkhe *)
Require Import Reals.

Record State := mkState {
  C : R;
  F : R;
  z : R
}.

Axiom conservation : forall s : State, (C s + F s = 1)%R.

Definition valid_state (s : State) : Prop :=
  (C s + F s = 1)%R /\ (0 <= z s <= 1)%R.
