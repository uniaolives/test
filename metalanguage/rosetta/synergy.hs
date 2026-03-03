module Synergy where

data AgentState = AgentState { info :: [Double] }
data Synergy = Emergence | Decoherence

mutualInfo :: AgentState -> AgentState -> Double
mutualInfo a b = 1.0 -- dummy

emerge :: AgentState -> AgentState -> Synergy
emerge stateA stateB
    | mutualInfo stateA stateB > 0.618 = Emergence
    | otherwise                            = Decoherence
