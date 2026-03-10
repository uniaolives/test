(ns arkhe.orb.temporal
  (:require [clojure.tools.logging :as log]))

(defn start-kuramoto-sync! [config lambda-fn]
  (log/info "Kuramoto Sync Agent active")
  true)

(defn stop-kuramoto! [agent]
  (log/info "Kuramoto Sync Agent stopped")
  true)
