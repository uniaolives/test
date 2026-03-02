::  /lib/arkhe/human-tool/hoon
::
::  Biblioteca para relação humano-ferramenta
::  Utiliza ponto fixo (base 100) para representar frações
::
|%
+$  human
  $:  processing-capacity=@ud      :: bits/min
      attention-span=@ud            :: minutes
      current-load=@ud              :: 0..100
      goals=(list @t)
  ==
+$  tool
  $:  output-volume=@ud             :: tokens/min
      output-entropy=@ud            :: bits/token (ponto fixo base 100)
      has-discernment=?             :: sempre %.n
      has-intentionality=?          :: sempre %.n
      has-perception=?              :: sempre %.n
  ==
+$  log-event
  $%  [%blocked reason=@t load=@ud]
      [%generated load=@ud intent=@t]
      [%reviewed approved=? output=@t]
  ==
++  new-guard
  |=  [h=human t=tool]
  ^-  (pair human tool (qeu log-event))
  [h t *qeu]
::
++  propose-interaction
  |=  [guard=(pair human tool (qeu log-event)) intent=@t]
  ^-  (unit [response=@t new-guard=(pair human tool (qeu log-event))])
  =/  h  -.guard
  =/  t  +.guard
  =/  log  +<.guard
  ::  load = (volume * entropy) / capacity
  ::  entropy já está em base 100, então load estará em base 100
  =/  load  (div (mul output-volume.t output-entropy.t) processing-capacity.h)
  ?:  (gth load 70)
    =.  log  (~(put to log) [%blocked 'cognitive-overload' load])
    `[~ [h t log]]
  ?:  (gth current-load.h 80)
    =.  log  (~(put to log) [%blocked 'human-overloaded' current-load.h])
    `[~ [h t log]]
  =/  response  (cat 3 'Generated content for: ' intent)
  =/  impact  (div (mul load 30) 100)
  =.  current-load.h  (min 100 (add current-load.h impact))
  =.  log  (~(put to log) [%generated load intent])
  `[response [h t log]]
::
++  review
  |=  [guard=(pair human tool (qeu log-event)) output=@t approved=?]
  ^-  (pair human tool (qeu log-event))
  =/  h  -.guard
  =/  t  +.guard
  =/  log  +<.guard
  =.  log  (~(put to log) [%reviewed approved output])
  ?:  approved
    =.  current-load.h  (max 0 (sub current-load.h 10))
    [h t log]
  [h t log]
--
