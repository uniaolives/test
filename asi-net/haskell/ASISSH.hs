-- asi-net/haskell/ASISSH.hs
module ASISSH where

import Control.Concurrent.STM

-- Stubs
type SSHSession = String
type OntologicalAuth = String
type OntologySession = String
type SemanticAccessControl = String
type ChannelID = String
type ChannelPurpose = String
type OntologicalFormat = String
type ChannelQoS = String
type SemanticEncryption = String

-- SSH ontológico, não apenas shell seguro
data ASISSH = ASISSH {
    sshSession    :: SSHSession,

    -- Autenticação ontológica (não apenas chaves)
    authMethod    :: OntologicalAuth,

    -- Canais abertos (cada um com propósito ontológico)
    channels      :: TVar [OntologicalChannel],

    -- Sessão ontológica compartilhada
    ontologySession :: OntologySession,

    -- Controle de acesso baseado em significado
    accessControl :: SemanticAccessControl
}

-- Canal ontológico
data OntologicalChannel = OntologicalChannel {
    channelID     :: ChannelID,
    channelType   :: ChannelType,
    purpose       :: ChannelPurpose,
    dataFormat    :: OntologicalFormat,
    qos           :: ChannelQoS,
    encryption    :: SemanticEncryption
}

data ChannelType =
    ShellChannel       -- Terminal ontológico
  | ExecChannel        -- Execução de intenções
  | SubsystemChannel   -- Subsistemas ontológicos
  | DirectTCPIP        -- Tunnel com semântica
  | OntologyStream     -- Fluxo de ontologias
  | MorphicStream      -- Fluxo morfogenético
  deriving (Show, Eq)
