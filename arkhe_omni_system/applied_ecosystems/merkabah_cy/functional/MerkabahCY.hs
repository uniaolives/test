-- MerkabahCY.hs
module MerkabahCY where

import System.Random

type CYVariety = (Int, Int, [Double], [Double])  -- h11, h21, metricDiag, complexModuli

createCY :: Int -> Int -> CYVariety
createCY h11 h21 = (h11, h21, replicate h11 1.0, replicate h21 0.0)

euler :: CYVariety -> Int
euler (h11, h21, _, _) = 2 * (h11 - h21)

complexityIndex :: CYVariety -> Double
complexityIndex (h11, _, _, _) = fromIntegral h11 / 491.0

-- MAPEAR_CY com recursão (simulado)
mapModuli :: CYVariety -> Int -> StdGen -> (CYVariety, StdGen)
mapModuli cy 0 g = (cy, g)
mapModuli (h11, h21, metric, moduli) n g =
    let (deltas, g') = generateDeltas h21 g
        newModuli = zipWith (+) moduli deltas
    in mapModuli (h11, h21, metric, newModuli) (n-1) g'

generateDeltas :: Int -> StdGen -> ([Double], StdGen)
generateDeltas n g =
    let (g1, g2) = split g
        deltas = take n (randomRs (-0.1, 0.1) g1)
    in (deltas, g2)

-- GERAR_ENTIDADE
generateEntity :: Int -> CYVariety
generateEntity seed =
    let rng = mkStdGen seed
        (h11, rng') = randomR (200, 491) rng
        (h21, _) = randomR (100, 400) rng'
    in createCY h11 h21
