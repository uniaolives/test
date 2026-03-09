-- OrbLogic.hs
module OrbLogic where

data Orb = Orb { stability :: Double, frequency :: Double } deriving (Show)

collapseWavefunction :: Double -> Double -> Maybe Orb
collapseWavefunction lambda freq
    | lambda > 0.618 = Just (Orb lambda freq)
    | otherwise      = Nothing

transmit :: Orb -> String -> Either String String
transmit orb handover
    | stability orb > 0.5 = Right ("Transmitted: " ++ handover)
    | otherwise           = Left "Wormhole collapsed"
