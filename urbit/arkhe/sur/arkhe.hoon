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
                    threshold=@ud
                    actual=@ud
                ==
+$  node        $:  id=@p
                    state=*
                    caps=(map term gate)
                    log=(list [time=@da intent=intent result=*])
                    coherence=@rs
                ==
--
