-- safecore-9d/frontend/love4d/oracle.lua
-- The Oracle Interface: Real-time constitutional guidance system
-- Integrates Prophet and Compiler to provide advice to the Architect

local Prophet = require("prophet")
local Compiler = require("compiler")

local Oracle = {}
Oracle.__index = Oracle

function Oracle.new()
    local self = setmetatable({
        prophet = Prophet.new(),
        compiler = Compiler.new(),
        history = {}
    }, Oracle)
    return self
end

function Oracle:query(timeline, question_type, params)
    if question_type == "SHOULD_I_CARVE" then
        return self:analyze_carving(timeline, params.carving)
    elseif question_type == "FUTURE_PROOF" then
        return self.prophet:predict_outcome(timeline, {})
    end
    return { advice = "The Oracle ponders the 33 dimensions in silence." }
end

function Oracle:analyze_carving(timeline, carving)
    local prediction = self.prophet:predict_outcome(timeline, {carving})
    local proof = self.compiler:compile_carving(carving, timeline)

    local recommendation = "ABSTAIN"
    if prediction.success_probability > 0.8 then
        recommendation = "CARVE"
    end

    local advice = {
        decision = recommendation,
        confidence = prediction.oracle_confidence,
        risk = prediction.collapse_risk,
        proof_status = proof.formal_proof.status
    }

    table.insert(self.history, advice)
    return advice
end

return Oracle
