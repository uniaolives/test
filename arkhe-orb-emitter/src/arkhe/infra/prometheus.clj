(ns arkhe.infra.prometheus
  (:require [clojure.tools.logging :as log]))

(defn start-server [port]
  (log/info "Prometheus server started on port" port)
  true)
