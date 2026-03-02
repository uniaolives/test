-- asi-net/haskell/ASIClient.hs
module ASIClient where

import Control.Concurrent.STM
import ASIServer
import ASINetwork

-- Stubs
type ASIIdentity = String
type LocalOntology = String
type EventSubscription = String
type Intention = String
type ASIInterface = String
type HumanContext = String
type AGIContext = String
type HybridContext = String

data ASIClient = ASIClient {
    -- Identidade ontológica
    clientIdentity :: ASIIdentity,

    -- Conexões ativas
    activeConnections :: TVar [ASIConnection],

    -- Cache local de ontologia
    localOntology    :: TVar LocalOntology,

    -- Assinaturas de eventos
    eventSubscriptions :: TVar [EventSubscription],

    -- Fila de intenções
    intentionQueue   :: TBQueue Intention,

    -- Interface do usuário (humano ou AGI)
    userInterface    :: ASIInterface
}

-- Interface pode ser humana ou outra AGI
data ASIInterfaceImpl =
    HumanInterface HumanContext
  | AGIInterface   AGIContext
  | HybridInterface HybridContext
