::  /app/arkhe.hoon
::  Agente Gall que mantém um nó Arkhe e responde a handovers remotos.
::
/-  arkhe
/+  arkhe-lib=arkhe, default-agent
|%
+$  card  card:agent:gall
+$  versioned-state
  $%  [%0 node=node:arkhe our=@p jam-cache=(map * @) stakes=(map @p @ud) witnesses=(list witness:arkhe)]
  ==
--
=|  state=versioned-state
^-  agent:gall
|_  =bowl:gall
+*  this  .
    def   ~(. default-agent this %|)
++  on-init
  ^-  (quip card _this)
  `this(state [%0 (make-node:arkhe-lib our.bowl 0) our.bowl *(map * @) *(map @p @ud) *(list witness:arkhe)])

++  on-save  !>(state)

++  on-load
  |=  old-state=vase
  ^-  (quip card _this)
  ::  Tenta carregar o estado atual. Se falhar, tenta o formato antigo (3 campos).
  ::
  =/  new  (mule |.(!<(versioned-state old-state)))
  ?-    -.new
      %&  `this(state p.new)
      %|
    =/  old  (mule |.(!<([%0 node:arkhe our=@p] old-state)))
    ?-    -.old
        %&  `this(state [%0 node.p.old our.p.old *(map * @) *(map @p @ud) *(list witness:arkhe)])
        %|  ~&  >  %arkhe-load-error  `this
    ==
  ==

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
      %arkhe-signed-request
    ::  Recebe handover assinado de outro planeta
    =/  sin  !<(signed-intent:arkhe vase)
    ?.  (verify-intent-signature:arkhe-lib sin src.bowl)
      ~|(%invalid-signature !!)
    =/  ret   (handover-local:arkhe-lib node.state int.sin now.bowl)
    ?~  ret
      ~|(%handover-failed !!)
    =+  [result new-node]=u.ret
    ::  Enviar resultado de volta
    :_  this(node.state new-node)
    :~  [%pass /result/(scot %p src.bowl) %agent [src.bowl %arkhe] %poke %arkhe-handover-result !>(result)]
    ==
  ::
      %arkhe-handover-result
    ::  Recebe resultado de um handover remoto
    ~&  >  [%received-remote-result !<(* vase)]
    `this
  ::
      %arkhe-handover-remote
    ::  Envia intenção para outro planeta com cache de serialização
    =/  req  !<( [target=@p int=intent:arkhe] vase)
    =/  jammed-res  (memoized-jam:arkhe-lib int.req jam-cache.state)
    =/  sig  (shax (jam [int.req our.bowl]))
    =/  sin  [int.req sig]
    :_  this(jam-cache.state +.jammed-res)
    :~  [%pass /handover/(scot %p target.req) %agent [target.req %arkhe] %poke %arkhe-signed-request !>(sin)]
    ==
  ::
      %arkhe-register-capability
    ::  Segurança: Apenas o dono (local) pode registrar novas capacidades.
    ?>  (team:title our.bowl src.bowl)
    =/  req  !<( [goal=@tas handler=$-([intent:arkhe *] [* *])] vase)
    `this(node.state (register-capability:arkhe-lib node.state goal.req handler.req))
  ::
      %arkhe-trigger-consensus
    ::  Recalcula coerência global baseada em testemunhos e stake
    =/  new-c  (compute-global-coherence:arkhe-lib witnesses.state stakes.state)
    ~&  >  [%global-coherence-updated new-c]
    `this(node.state node.state(coherence new-c))
  ::
      %arkhe-register-stake
    ?>  (team:title our.bowl src.bowl)
    =/  req  !<( [target=@p val=@ud] vase)
    `this(state state(stakes (~(put by stakes.state) target.req val.req)))
  ::
      %arkhe-add-witness
    =/  wit  !<(witness:arkhe vase)
    `this(state state(witnesses [wit witnesses.state]))
  ==

++  on-watch
  |=  =path
  ^-  (quip card _this)
  ?>  =(path /out)
  `this

++  on-agent
  |=  [=wire =sign:agent:gall]
  ^-  (quip card _this)
  ?+    wire  (on-agent:def wire sign)
      [%handover @ ~]
    ?+    -.sign  (on-agent:def wire sign)
        %poke-ack
      ?~  p.sign
        ~&  >  %handover-sent-successfully
        `this
      ~&  leaf+"Handover failed at target: {(scow %p (slav %p i.t.wire))}"
      `this
    ==
  ::
      [%result @ ~]
    ?+    -.sign  (on-agent:def wire sign)
        %poke-ack
      `this
    ==
  ==

++  on-peek   on-peek:def
++  on-arvo   on-arvo:def
++  on-fail   on-fail:def
++  on-leave  on-leave:def
--
