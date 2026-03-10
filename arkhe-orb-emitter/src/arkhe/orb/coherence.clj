(ns arkhe.orb.coherence
  (:require [clojure.math :as math]))

(defn tunneling-probability [lambda-2 delta-t-ms]
  (let [barrier (* delta-t-ms (- 1.0 lambda-2))]
    (math/exp (* -2.0 barrier))))

(defn quantize-levels [n-levels gap]
  (map (fn [n] {:n (inc n) :energy (* (inc n) gap)}) (range n-levels)))
