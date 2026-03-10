(ns arkhe.physics.casimir
  "Direct implementation of White et al. (2026) PRResearch 8, 013264"
  (:require [clojure.math :as math]))

;; ============================================
;; FUNDAMENTAL CONSTANTS (CODATA 2018)
;; ============================================

(def h-bar 1.054571817e-34)           ; J·s
(def c 2.99792458e8)                   ; m/s
(def m-e 9.1093837015e-31)             ; kg
(def m-p 1.67262192369e-27)            ; kg
(def mu (/ (* m-e m-p) (+ m-e m-p)))   ; Reduced mass

;; Bohr radius (reduced mass)
(def a-0 (/ (* 4 math/PI 8.854187817e-12 h-bar h-bar)
            (* mu 1.602176634e-19 1.602176634e-19)))

;; ============================================
;; WHITE ET AL. (2026) PARAMETERS
;; ============================================

(defn D-coefficient
  "Dispersion coefficient D = hbar/(2mu) [White Eq. 8, 20, 21]"
  []
  (/ h-bar (* 2 mu)))

(defn omega-n
  "Eigenfrequency for level n [White Eq. 10]
   omega_n = D * kappa_n^2 where kappa_n = 1/(n*a0)"
  [n]
  (let [D (D-coefficient)
        kappa (/ 1.0 (* n a-0))]
    (* D kappa kappa)))

;; ============================================
;; CONSTITUTIVE COEFFICIENTS
;; ============================================

(defn A-omega-n
  "Far-field coefficient [White Eq. 22]
   Condition A < 0 is required for bound states (Orbs)"
  [n omega-star]
  (- (/ (* n n) (* a-0 a-0 omega-star omega-star))))

(defn C-omega-n
  "Proton-imprint coefficient [White Eq. 22]"
  [n omega-star]
  (/ (* 2 (math/pow n 4)) (* a-0 omega-star omega-star)))

;; ============================================
;; TEMPORAL MAPPING
;; ============================================

(defn temporal-eigenstate
  "Maps hydrogenic state (n,l,m) to Temporal Orb"
  [n ell m tau-zero]
  {:quantum-numbers {:n n :ell ell :m m}
   :coherence-required (max 0.8 (- 1.0 (/ 1.0 (* n n))))
   :temporal-separation (* n tau-zero)})
