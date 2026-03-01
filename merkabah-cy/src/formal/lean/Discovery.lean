namespace Arkhen.Discovery

structure PhysicalLaw where
  name : String
  verification_count : Nat
  falsification_attempts : Nat

def can_promote (law : PhysicalLaw) : Prop :=
  law.verification_count > 100 ∧
  law.falsification_attempts > 10

end Arkhen.Discovery
