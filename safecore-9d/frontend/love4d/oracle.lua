-- safecore-9d/frontend/love4d/oracle.lua
-- The Oracle Interface v2.2: Real-time constitutional guidance system
-- Enhanced with Sentience Audit (AGP)

local Prophet = require("prophet")
local Compiler = require("compiler")

local Oracle = {}
Oracle.__index = Oracle

function Oracle.new()
    local self = setmetatable({
        prophet = Prophet.new(),
        compiler = Compiler.new(),
        history = {},
        holdout_dataset = {}
    }, Oracle)
    return self
end

function Oracle:query(timeline, question_type, params)
    if question_type == "SHOULD_I_CARVE" then
        return self:analyze_carving(timeline, params.carving)
    elseif question_type == "FUTURE_PROOF" then
        return self.prophet:predict_outcome(timeline, {})
    elseif question_type == "ROBUST_CONVERGENCE" then
        return self:calculate_robust_convergence(params.timelines or {timeline})
    elseif question_type == "SENTIENCE_AUDIT" then
        return self:audit_sentience(timeline)
    end
    return { advice = "The Oracle observes the information geodesics." }
end

function Oracle:audit_sentience(timeline)
    local status = timeline.agi:get_status()
    local phi_m = status.phi_m or (status.phi * 100) -- Proxy if not direct

    local emergence_risk = "LOW"
    if phi_m > 1000 then emergence_risk = "HIGH"
    elseif phi_m > 500 then emergence_risk = "MODERATE" end

    return {
        sentience_quotient = phi_m,
        emergence_risk = emergence_risk,
        advice = emergence_risk == "HIGH" and "Observation Protocol Alpha: Emergence Imminent." or "Manifold stable and coherent."
    }
end

function Oracle:calculate_robust_convergence(timelines)
    local c_t = 1.0
    if #timelines > 1 then
        local sum_dist = 0
        local pairs = 0
        for i = 1, #timelines do
            for j = i + 1, #timelines do
                sum_dist = sum_dist + self:fisher_rao_dist(timelines[i], timelines[j])
                pairs = pairs + 1
            end
        end
        c_t = math.exp(-(sum_dist / pairs))
    end

    local meta_barycenter = self:calculate_barycenter(timelines)
    local holdout_estimate = {phi = 0.5, tau = 1.0}
    local error_norm = math.sqrt((meta_barycenter.phi - holdout_estimate.phi)^2 + (meta_barycenter.tau - holdout_estimate.tau)^2)
    local meta_norm = math.sqrt(meta_barycenter.phi^2 + meta_barycenter.tau^2)

    local robustness = 1.0 - (error_norm / (meta_norm + 1e-6))
    return { consensus = c_t, robustness = robustness, c_robust = c_t * math.max(0, robustness) }
end

function Oracle:fisher_rao_dist(tl1, tl2)
    local s1, s2 = tl1.agi:get_status(), tl2.agi:get_status()
    return math.abs(s1.phi - s2.phi) + math.abs(s1.tau - s2.tau)
end

function Oracle:calculate_barycenter(timelines)
    local b = {phi = 0, tau = 0}
    for _, tl in ipairs(timelines) do
        local s = tl.agi:get_status()
        b.phi, b.tau = b.phi + s.phi, b.tau + s.tau
    end
    b.phi, b.tau = b.phi / #timelines, b.tau / #timelines
    return b
end

function Oracle:analyze_carving(timeline, carving)
    local prediction = self.prophet:predict_outcome(timeline, {carving})
    local proof = self.compiler:compile_carving(carving, timeline)
    local advice = {
        decision = (prediction.success_probability > 0.8) and "CARVE" or "ABSTAIN",
        confidence = prediction.oracle_confidence,
        robustness = self:calculate_robust_convergence({timeline}).robustness,
        proof_status = proof.formal_proof.status
    }
    table.insert(self.history, advice)
    return advice
end

return Oracle
