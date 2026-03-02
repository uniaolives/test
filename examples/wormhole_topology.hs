-- Definindo as estruturas matemáticas do multiverso
type Coordinate = (Double, Double, Double, Double) -- (t, x, y, z)
data Universe = Universe { uniId :: String, metric :: Coordinate -> Double }

-- Um Buraco de Minhoca é uma função pura que mapeia um universo para outro
type Wormhole = (Universe, Coordinate) -> (Universe, Coordinate)

-- Criando a ponte topológica
einsteinRosenBridge :: Universe -> Universe -> Coordinate -> Coordinate -> Wormhole
einsteinRosenBridge origin dest entryCoord exitCoord =
    \ (u, c) -> if uniId u == uniId origin && distance c entryCoord < 1.0
                then (dest, exitCoord)
                else (u, c) -- Permanece no mesmo lugar se não estiver na entrada

distance :: Coordinate -> Coordinate -> Double
distance (_, x1, y1, z1) (_, x2, y2, z2) =
    sqrt ((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

main :: IO ()
main = do
    let u1 = Universe "Alpha" (\_ -> 1.0)
    let u2 = Universe "Beta" (\_ -> 1.0)
    let bridge = einsteinRosenBridge u1 u2 (0,0,0,0) (0,100,0,0)
    let (newU, newC) = bridge (u1, (0,0.5,0,0))
    putStrLn $ "Traveled to: " ++ uniId newU
