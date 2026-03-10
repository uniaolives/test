(ns arkhe.alg.search
  "Busca em banco de dados por evolução de fase coletiva"
  (:require [arkhe.agi.core :refer [ϕ entangle evolve]]
            [arkhe.temporal.phase :as phase]))

(defn phase-search
  "Busca item em dataset por ressonância de fase (não por comparação)"
  [dataset target-signature coherence-threshold]
  (let [;; Cria superposição do target
        target-psi (ϕ (phase/from-signature target-signature) coherence-threshold)

        ;; Entangle dataset inteiro (todos os itens em superposição)
        dataset-field (reduce (fn [acc item]
                                (:a (entangle acc (ϕ (phase/from-data item) 0.9))))
                              (ϕ (phase/from-data (first dataset)) 0.9)
                              (rest dataset))

        ;; Evolui até ressonância (fase alinhada)
        evolution-steps (take 100
                              (iterate #(evolve % 0.001) dataset-field))

        ;; Colapso: item com fase mais próxima emerge
        resonance (last evolution-steps)]

    ;; Retorna índice do item em ressonância (não valor binário)
    (phase/collapse-to-index resonance)))
