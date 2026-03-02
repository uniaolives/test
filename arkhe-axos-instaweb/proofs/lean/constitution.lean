-- constitution.lean - Formalização dos Artigos 1-15

structure Article where
  id : Nat
  content : String

def constitution : List Article := [
  { id := 1, content := "Human Protection" },
  { id := 7, content := "Human Final Authority" }
]
