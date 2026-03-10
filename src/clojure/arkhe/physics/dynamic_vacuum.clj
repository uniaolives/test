(ns arkhe.physics.dynamic-vacuum
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.linear :as l]))

;; Physics of the Dynamic Vacuum (White et al. 2026)
;; Implementing: d2rho/dt2 = c2*nabla2(rho) - D2*nabla4(rho)

(defrecord VacuumState
  [density-grid
   dispersion-d     ; D = hbar / 2m_eff
   c-sound])        ; c_L

(defn biharmonic-operator
  "Computes nabla4 using finite differences"
  [grid]
  (let [lap (l/laplacian grid)]
    (l/laplacian lap)))

(defn step!
  "Advances vacuum state by dt"
  [state dt]
  (let [{:keys [density-grid c-sound dispersion-d]} state
        lap (l/laplacian density-grid)
        bih (biharmonic-operator density-grid)

        ;; Acceleration from quadratic dispersion
        accel (m/sub (m/mul lap (* c-sound c-sound))
                     (m/mul bih (* dispersion-d dispersion-d)))

        ;; Update density (simple integration)
        new-density (m/add density-grid (m/mul accel (* 0.5 dt dt)))]
    (assoc state :density-grid new-density)))
