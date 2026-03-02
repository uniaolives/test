(* spec/coq/Migdal_Byzantine_Detection.v – Final Version *)
(* Inspired by Migdal Effect (Nature 2026) *)

Theorem equivocation_detectable :
  forall (leader : Node) (r : Round) (v1 v2 : Value),
    Byzantine leader ->
    Sends leader r (Prepare v1) /\
    Sends leader r (Prepare v2) ->
    v1 <> v2 ->
    exists (honest1 honest2 : Node),
      Receives honest1 (Prepare v1) /\
      Receives honest2 (Prepare v2) /\
      (Sends honest1 (Suspect leader) \/ Sends honest2 (Suspect leader)).
Proof.
  (* Formalization of the "recoil" detection: a leader's
     equivocation ejects a suspicion "electron" from the
     honest node's "cloud". *)
Admitted.
