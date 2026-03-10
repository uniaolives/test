(ns arkhe.orb.http4
  (:require [clojure.tools.logging :as log]))

(defn start-server! [config deps]
  (log/info "HTTP/4 Server started on port" (:port config))
  true)

(defn stop-server! [server]
  (log/info "HTTP/4 Server stopped")
  true)
