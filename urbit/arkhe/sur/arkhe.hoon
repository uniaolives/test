|%
+$  intent      $:  goal=@tas
                    constraints=(list constraint)
                    metrics=(list metric)
                ==
+$  constraint  $:  type=@tas           ::  'time', 'energy', 'cost', etc.
                    operator=@tas        ::  'lt', 'le', 'eq', 'gt', 'ge'
                    value=@ud
                ==
+$  metric      $:  name=@tas
                    threshold=@rs       :: Coerência requer ponto flutuante
                    actual=@rs
                ==
+$  node        $:  id=@p
                    state=*
                    caps=(map term gate)
                    log=(list [time=@da intent=intent result=*])
                    coherence=@rs
                ==
::  Novos tipos para Arkhe(n) Otimizado (Ω+∞+257)
::
+$  stake       @p                      :: ship com participação garantida
+$  witness     $:  observer=@p
                    observed=@p
                    value=@rs           :: Coerência observada
                    timestamp=@da
                ==
+$  signed-intent  [int=intent sig=@uvH]
--
