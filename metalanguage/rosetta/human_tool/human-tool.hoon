::  /lib/arkhe/human-tool/hoon
::
::  Biblioteca para relação humano-ferramenta
::
|%
+$  human
  $:  processing-capacity=@ud      :: bits/min
      attention-span=@ud            :: minutes
      current-load=@ud
      goals=(list @t)
  ==
+$  tool
  $:  output-volume=@ud             :: tokens/min
      output-entropy=@ud            :: bits/token
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
::  ++  propose-interaction
++  propose-interaction
  |=  [guard=(pair human tool (qeu log-event)) intent=@t]
  ^-  (unit [response=@t new-guard=(pair human tool (qeu log-event))])
  =/  h  -.guard
  =/  t  +.guard
  =/  log  +<.guard
  =/  load  (div (mul output-volume.t output-entropy.t) processing-capacity.h)
  ?:  (gth load 0.7)
    =.  log  (~(put to log) [%blocked 'cognitive-overload' load])
    `[~ [h t log]]
  ?:  (gth current-load.h 0.8)
    =.  log  (~(put to log) [%blocked 'human-overloaded' current-load.h])
    `[~ [h t log]]
  =/  response  "Generated content for: {intent}"
  =/  impact  (mul load 0.3)
  =.  current-load.h  (min 1.0 (add current-load.h impact))
  =.  log  (~(put to log) [%generated load intent])
  `[response [h t log]]
::  ++  review
++  review
  |=  [guard=(pair human tool (qeu log-event)) output=@t approved=?]
  ^-  (pair human tool (qeu log-event))
  =/  h  -.guard
  =/  t  +.guard
  =/  log  +<.guard
  =.  log  (~(put to log) [%reviewed approved output])
  ?:  approved
    =.  current-load.h  (max 0 (sub current-load.h 0.1))
    [h t log]
  [h t log]
--
