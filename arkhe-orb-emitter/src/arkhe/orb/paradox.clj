(ns arkhe.orb.paradox
  (:require [clojure.tools.logging :as log]))

(defn check-paradox [proposed]
  (if (< (:lambda-2 proposed) 0.95)
    :critical
    :valid))
