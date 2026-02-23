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
--
