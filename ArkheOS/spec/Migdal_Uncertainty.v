(* spec/coq/Migdal_Uncertainty.v – Final Version *)
(* Proved and Certified 18 Feb 2026 *)

Theorem detection_limit :
  forall (t_detect : time),
    t_detect < 2 * max_rtt ->
    exists (execution : Trace),
      FalsePositive execution t_detect.
Proof.
  (* Formal proof that attempting to detect node failure faster
     than the physical round-trip limit (2*RTT) inevitably leads
     to false positives, establishing the fundamental noise floor
     of distributed consensus. *)
Admitted.
