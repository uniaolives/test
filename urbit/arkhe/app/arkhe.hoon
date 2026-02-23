::  /app/arkhe.hoon
::  Agente Gall que mantém um nó Arkhe e responde a handovers remotos.
::
/-  arkhe
/+  arkhe-lib=arkhe, default-agent
|%
+$  card  card:agent:gall
+$  versioned-state
  $%  [%0 node=node:arkhe our=@p]
  ==
--
=|  state=versioned-state
^-  agent:gall
|_  =bowl:gall
+*  this  .
    def   ~(. default-agent this %|)
++  on-init
  ^-  (quip card _this)
  `this(state [%0 (make-node:arkhe-lib our.bowl 0) our.bowl])

++  on-save  !>(state)

++  on-load
  |=  old-state=vase
  ^-  (quip card _this)
  =/  old  !<(versioned-state old-state)
  `this(state old)

++  on-poke
  |=  [=mark =vase]
  ^-  (quip card _this)
  ?>  ?=(%0 -.state)
  ?+    mark    (on-poke:def mark vase)
      %arkhe-handover-request
    =/  data  !<(intent:arkhe vase)
    =/  ret   (handover-local:arkhe-lib node.state data now.bowl)
    ?~  ret
      ~|(%handover-failed !!)
    =+  [result new-node]=u.ret
    :-  ~[[%give %fact ~[/out] %arkhe-handover-result !>(result)]]
    this(node.state new-node)
  ::
      %arkhe-register-capability
    ::  Segurança: Apenas o dono (local) pode registrar novas capacidades.
    ?>  (team:title our.bowl src.bowl)
    =/  req  !<( [goal=@tas handler=$-([intent:arkhe *] [* *])] vase)
    `this(node.state (register-capability:arkhe-lib node.state goal.req handler.req))
  ==

++  on-watch
  |=  =path
  ^-  (quip card _this)
  ?>  =(path /out)
  `this

++  on-peek   on-peek:def
++  on-agent  on-agent:def
++  on-arvo   on-arvo:def
++  on-fail   on-fail:def
++  on-leave  on-leave:def
--
