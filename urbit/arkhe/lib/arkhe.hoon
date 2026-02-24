::  /lib/arkhe.hoon
::  Biblioteca principal para a Arkhe(n) Language em Urbit.
::
/-  arkhe
|%
::  ++  make-node  : cria um novo nó com estado inicial e log vazio.
::
++  make-node
  |=  [id=@p initial-state=*]
  ^-  node:arkhe
  :*  id
      initial-state
      *(map term gate)
      *(list [time=@da intent=intent:arkhe result=*])
      .1.0                     :: coerência inicial (máxima)
  ==

::  ++  register-capability  : associa uma intenção (goal) a um handler.
::
++  register-capability
  |=  [n=node:arkhe goal=@tas handler=$-([intent:arkhe *] [* *])]
  ^-  node:arkhe
  n(caps (~(put by caps.n) goal handler))

::  ++  handover-local  : executa um handover localmente (no mesmo nó).
::
++  handover-local
  |=  [n=node:arkhe incoming=intent:arkhe now=@da]
  ^-  (unit [result=* n=node:arkhe])
  =/  handler  (~(get by caps.n) goal.incoming)
  ?~  handler  ~
  =/  valid=?
    ?.  ?=(@ud state.n)  &
    (check-constraints constraints.incoming state.n)
  ?.  valid  ~
  =/  ret  (u.handler incoming state.n)
  =/  new-node  n(state +.ret)
  =/  entry  [now intent=incoming result=-.ret]
  `[-.ret new-node(log [entry log.new-node])]

::  ++  check-constraints  : verifica se uma intenção satisfaz as constraints
::
++  check-constraints
  |=  [cons=(list constraint:arkhe) state-val=@ud]
  ^-  ?
  %+  levy  cons
  |=  c=constraint:arkhe
  ?+    operator.c  ~|(unknown-operator+operator.c !!)
    %lt  (lth state-val value.c)
    %le  (lte state-val value.c)
    %eq  =(state-val value.c)
    %gt  (gth state-val value.c)
    %ge  (gte state-val value.c)
  ==

::  ++  update-coherence  : recalcula a coerência do nó baseado no log.
::
++  update-coherence
  |=  n=node:arkhe
  ^-  @rs
  ::  TODO: Implementar contagem real de falhas baseada em métricas
  =/  failures  0
  ?:  =(0 (lent log.n))  .1.0
  (quo:rs .1.0 (sun:rs (add 1 failures)))

::  ++  memoized-jam  : serialização otimizada com cache.
::
++  memoized-jam
  |=  [a=* cache=(map * @)]
  ^-  [@ (map * @)]
  =/  cached  (~(get by cache) a)
  ?~  cached
    =/  jammed  (jam a)
    [jammed (~(put by cache) a jammed)]
  [u.cached cache]

::  ++  verify-intent-signature  : verifica integridade e autoria da intenção.
::
++  verify-intent-signature
  |=  [sin=signed-intent:arkhe ship=@p]
  ^-  ?
  ::  Protótipo: verifica hash da intenção com o ship.
  ::  Em prod usaríamos ed25519 via jael.
  =(sig.sin (shax (jam [int.sin ship])))

::  ++  compute-global-coherence  : consenso sobre coerência global.
::
++  compute-global-coherence
  |=  [wits=(list witness:arkhe) stakes=(map @p @ud)]
  ^-  @rs
  ::  1. Filtrar witnesses de ships sem stake
  =/  filtered  (skim wits |=(w=witness:arkhe (~(has by stakes) observer.w)))
  ?:  =(0 (lent filtered))  .0.0
  ::  2. Ordenar por valor
  =/  sorted
    %+  sort  filtered
    |=  [a=witness:arkhe b=witness:arkhe]
    (lth:rs value.a value.b)
  ::  3. Média aparada (Trimmed Mean) - remove 10% de cada extremidade
  =/  len  (lent sorted)
  =/  trim  (div len 10)
  =/  trimmed  (scag (sub len (add trim trim)) (slag trim sorted))
  ?:  =(0 (lent trimmed))  .0.0
  ::  4. Média ponderada pelo stake
  =/  total-weight  0
  =/  weighted-sum  .0.0
  |-
  ?~  trimmed  (quo:rs weighted-sum (sun:rs (max 1 total-weight)))
  =/  s  (~(gut by stakes) observer.i.trimmed 1)
  $(trimmed t.trimmed, total-weight (add total-weight s), weighted-sum (add:rs weighted-sum (mul:rs value.i.trimmed (sun:rs s))))
--
