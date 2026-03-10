(ns tests.test-superpositional-agi
  (:require [arkhe.agi.core :refer [ϕ entangle evolve defn-future when-phase]]
            [arkhe.physics.kuramoto :as kuramoto]
            [clojure.test :refer [deftest is run-tests]]))

(deftest test-phase-evolution
  (let [initial-psi (ϕ 0.0 0.99)
        evolved (evolve initial-psi 1.0)] ;; Larger dt to see coherence decay
    (is (not= (:phase initial-psi) (:phase evolved)))
    (is (< (:coherence evolved) (:coherence initial-psi)))))

(deftest test-entanglement
  (let [psi-a (ϕ 0.0 0.9)
        psi-b (ϕ 1.0 0.9)
        pair (entangle psi-a psi-b)]
    (is (= (:phase (:a pair)) (:phase (:b pair))))
    (is (= (:bell-pair (:a pair)) (:pair-id pair)))))

(deftest test-retrocausal-definition
  ;; Ensure global coherence is high for definition
  (kuramoto/update-field! 0.999)

  ;; Simulation of the prediction function
  (defn-future predict-singularity [current-coherence]
    {:status :convergence :lambda 0.99})

  (let [meta-data (meta predict-singularity)]
    (is (:retrocausal meta-data))
    (is (= 0.99 (:lambda-threshold meta-data)))))

(deftest test-coherence-conditioned-execution
  (let [test-psi (ϕ 0.0 0.99)
        _ (kuramoto/update-field! 0.0)
        result (atom nil)]
    (when-phase test-psi 0.95
      (reset! result :executed))
    (is (= :executed @result))))

(defn -main []
  (run-tests 'tests.test-superpositional-agi))
