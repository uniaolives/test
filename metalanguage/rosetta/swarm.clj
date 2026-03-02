(defn calculate-synergy [agents]
  (reduce + (map :performance agents)))
