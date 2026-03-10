(ns arkhe.orb.emitter
  (:require [clojure.tools.logging :as log]
            [arkhe.orb.http4 :as http4]
            [arkhe.orb.temporal :as temporal]
            [arkhe.infra.datomic :as datomic]
            [arkhe.infra.prometheus :as prom])
  (:gen-class))

(def config
  {:http4 {:port 8080}
   :datomic {:uri "datomic:mem://arkhe-orb-emitter"}})

(defn -main [& args]
  (log/info "Starting Arkhe(n) Orb Emitter Pilot...")
  (prom/start-server 9090)
  (let [conn (datomic/connect! (:datomic config))]
    (datomic/init-schema! conn)
    (http4/start-server! (:http4 config) {:datomic conn})
    (temporal/start-kuramoto-sync! {} (fn [] 0.95))
    (log/info "System Ready.")))
