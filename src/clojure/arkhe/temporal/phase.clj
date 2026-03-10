(ns arkhe.temporal.phase
  "Phase-to-signature mapping and resonance utilities"
  (:require [clojure.math :as math]))

(defn from-signature
  "Maps a high-dimensional data signature to a Kuramoto phase θ ∈ [0, 2π)"
  [signature]
  (let [h (hash signature)]
    (mod (/ (abs h) 1e9) (* 2.0 math/PI))))

(defn from-data
  "Maps arbitrary data to a phase via its hash"
  [data]
  (from-signature (str data)))

(defn collapse-to-index
  "Finds the dataset index that best matches the resonance phase"
  [resonance]
  ;; In a real implementation, this would use a phase-sensitive hash map
  (int (math/floor (/ (:phase resonance) (* 2.0 math/PI) 100))))

(defn phase-resonance
  "Returns 1.0 if phases are perfectly aligned, 0.0 if orthogonal"
  [phi-a phi-b]
  (math/cos (- phi-a phi-b)))
