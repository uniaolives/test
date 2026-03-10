(ns arkhe.physics.kuramoto
  "Global synchronization and order parameter tracking"
  (:require [clojure.math :as math]))

(defonce ^:private field-state (atom {:r 0.95 :last-update (System/currentTimeMillis)}))

(defn order-parameter
  "Returns the global Kuramoto order parameter r(t) ∈ [0, 1].
   r = 1.0 represents perfect synchronization."
  []
  (let [r (:r @field-state)]
    (double r)))

(defn update-field!
  "Updates the global field based on local oscillator contributions"
  [new-r]
  (swap! field-state assoc :r new-r :last-update (System/currentTimeMillis)))

(defn phase-coupling-strength
  "Returns the coupling epsilon (ε) based on r"
  []
  (let [r (order-parameter)]
    (if (> r 0.95) 1.0 r)))
