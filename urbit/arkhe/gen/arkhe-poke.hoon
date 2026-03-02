::  /gen/arkhe-poke/hoon
::  Envia uma intenção para o agente Arkhe (local ou remoto).
::
::  Uso Local:  +arkhe-poke %increment
::  Uso Remoto: +arkhe-poke %increment ~marzod
::
/-  arkhe
:-  %say
|=  [[* eny=@uvJ *] [goal=@tas target=(unit @p) ~] *]
^-  cage
=/  int=intent:arkhe  [goal=goal constraints=~ metrics=~]
?~  target
  [%arkhe-handover-request !>(int)]
[%arkhe-handover-remote !>([target=u.target int=int])]
