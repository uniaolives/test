-- asi-net/haskell/ASIWebhooks.hs
module ASIWebhooks where

import ASINetwork

-- Stubs
type WebhookID = String
type TriggerCondition = String
type PayloadTransform = String
type RetryPolicy = String
type DigitalSignature = String
type EventType = String
type SemanticFilter = String
type ASIMessage = String
type OntologyUpdate = String
type Intention = String
type MorphicPattern = String
type SemanticMessage = String
type NodeTransformation = String

-- Webhook não é apenas HTTP callback
-- É uma reação ontológica a eventos
data OntologicalWebhook = OntologicalWebhook {
    hookID        :: WebhookID,

    -- Padrão de evento (ontológico, não apenas string)
    eventPattern  :: EventPattern,

    -- Ação a ser executada
    action        :: OntologicalAction,

    -- Condições de disparo
    conditions    :: [TriggerCondition],

    -- Transformação do payload
    transformation :: Maybe PayloadTransform,

    -- Políticas de retry (com backoff ontológico)
    retryPolicy   :: RetryPolicy,

    -- Assinatura do webhook
    signature     :: DigitalSignature
}

-- Padrão de evento ontológico
data EventPattern = EventPattern {
    eventType     :: EventType,
    sourceOntology :: OntologyType,
    targetOntology :: Maybe OntologyType,
    minimumCertainty :: Float,
    semanticFilter  :: SemanticFilter
}

-- Ação ontológica
data OntologicalAction =
    SendMessage ASIMessage
  | UpdateOntology OntologyUpdate
  | TriggerIntention Intention
  | MorphicResonance MorphicPattern
  | SemanticBroadcast SemanticMessage
  | CreateEdge EdgeRelation
  | TransformNode NodeTransformation
