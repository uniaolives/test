(ns arkhe.infra.datomic
  (:require [datomic.api :as d]
            [clojure.tools.logging :as log]))

(defn connect! [uri]
  (log/info "Connecting to Datomic:" uri)
  ;; d/connect or create mock
  nil)

(defn init-schema! [conn]
  (log/info "Datomic schema initialized")
  true)

(defn release! [conn]
  true)
