;; regras_arc.metta
;; Transformações geométricas
(= (rotate-90 $grid) (apply-rotation $grid 90))
(= (reflect $grid)   (apply-reflection $grid))

;; Composição de duas transformações
(= (compose $f $g $grid) ($f ($g $grid)))

;; Regra de descoberta de padrões
(= (find-pattern $grid) (match-arc-pattern $grid (memory-query $grid)))
