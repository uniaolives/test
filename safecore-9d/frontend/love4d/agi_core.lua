-- safecore-9d/frontend/love4d/agi_core.lua
-- Lua-AGI/ASI: Artificial Geometric Intelligence Core
-- Refined with K-FAC Inspired Adaptive Fisher Estimation and Geodesic Pruning

local AGI = {}

-- =========================================================================
-- 1. STATE SPACE GEOMETRY ENGINE (SSGE)
-- =========================================================================
local StateSpace = {}
StateSpace.__index = StateSpace

local PHI_THRESHOLD = 0.80
local TAU_MAX = 1.35
local MAX_DIMENSIONS = 33
local SPARSITY_BUDGET = 0.014   -- SOP-DA-01 1.4% rule

function StateSpace.new(dimensions)
    local self = setmetatable({}, StateSpace)
    self.dimensions = math.min(dimensions or 8, MAX_DIMENSIONS)
    self.coordinates = {}
    self.history = {}
    self.max_history = 5

    -- Adaptive Fisher Information Matrix (Diagonal approximation)
    self.fisher_diag = {}
    for i = 1, self.dimensions do
        self.coordinates[i] = 0.0
        self.fisher_diag[i] = 1.0 -- Identity baseline
    end

    self.step_count = 0
    self.cumulative_regret = 0.0
    self.phi = 0.0
    self.tau = 0.0
    self.mode = 0

    return self
end

-- Natural Gradient Update with Robbins-Monro adaptive Fisher estimation
function StateSpace:natural_update(gradient, learning_rate)
    self.step_count = self.step_count + 1
    learning_rate = learning_rate or 0.1

    -- 1. Adaptive Fisher Update (Bias mitigation)
    local alpha = math.pow(self.step_count, -0.6)
    for i = 1, self.dimensions do
        local g_sq = (gradient[i] or 0)^2
        self.fisher_diag[i] = (1 - alpha) * self.fisher_diag[i] + alpha * g_sq + 1e-6 -- Tikhonov regularizer
    end

    -- 2. Greedy Forward Selection (Sparsity Pruning)
    local importance = {}
    for i = 1, self.dimensions do
        importance[i] = { val = math.abs(gradient[i] or 0) / math.sqrt(self.fisher_diag[i]), idx = i }
    end
    table.sort(importance, function(a, b) return a.val > b.val end)

    local k = math.max(1, math.floor(self.dimensions * SPARSITY_BUDGET))
    local active_indices = {}
    for i = 1, k do active_indices[importance[i].idx] = true end

    -- 3. The Geodesic Step (Natural Gradient)
    for i = 1, self.dimensions do
        if active_indices[i] then
            local natural_grad = (gradient[i] or 0) / self.fisher_diag[i]
            self.coordinates[i] = self.coordinates[i] - learning_rate * natural_grad
        end
    end

    self:update_metrics()
end

function StateSpace:update_metrics()
    self:record_history()
    self:update_torsion()
    self:calculate_phi()
    self:update_mode()
end

function StateSpace:record_history()
    local current = {}
    for i = 1, self.dimensions do current[i] = self.coordinates[i] end
    table.insert(self.history, current)
    if #self.history > self.max_history then table.remove(self.history, 1) end
end

function StateSpace:update_torsion()
    if #self.history < 3 then self.tau = 0.0 return end
    local s1, s2, s3 = self.history[#self.history-2], self.history[#self.history-1], self.history[#self.history]
    local v1, v2 = {}, {}
    for i = 1, self.dimensions do
        v1[i] = (s2[i] or 0) - (s1[i] or 0)
        v2[i] = (s3[i] or 0) - (s2[i] or 0)
    end

    local cross_mag = 0.0
    if self.dimensions >= 3 then
        local cx = v1[2]*v2[3] - v1[3]*v2[2]
        local cy = v1[3]*v2[1] - v1[1]*v2[3]
        local cz = v1[1]*v2[2] - v1[2]*v2[1]
        cross_mag = math.sqrt(cx*cx + cy*cy + cz*cz)
    end

    local norm = 0
    for i = 1, self.dimensions do norm = norm + (v1[i]^2 + v2[i]^2) end
    self.tau = (norm > 0) and (cross_mag / math.sqrt(norm)) or 0.0
end

function StateSpace:calculate_phi()
    -- Phi via manifold volume proxy
    local det = 1.0
    for i = 1, self.dimensions do det = det * self.fisher_diag[i] end
    self.phi = math.min(1.0, math.log10(det + 1) / self.dimensions)
end

function StateSpace:update_mode()
    if self.phi < 0.2 then self.mode = 0
    elseif self.phi < 0.6 then self.mode = 1
    elseif self.phi < PHI_THRESHOLD then self.mode = 2
    else self.mode = 3 end
end

-- =========================================================================
-- 4. MAIN CONTROLLER
-- =========================================================================
AGI.__index = AGI

function AGI.new(config)
    local self = setmetatable({}, AGI)
    config = config or {}
    self.dimensions = config.dimensions or 33
    self.state = StateSpace.new(self.dimensions)
    self.status = "OPERATIONAL"
    return self
end

function AGI:initialize() return true end

function AGI:cycle(input_gradient)
    self.state:natural_update(input_gradient, 0.05)

    local error_mag = 0
    for i = 1, self.dimensions do error_mag = error_mag + (input_gradient[i] or 0)^2 end
    self.state.cumulative_regret = self.state.cumulative_regret + math.sqrt(error_mag)

    if self.state.mode == 3 then self.status = "HARD_FREEZE" end

    return {
        state = { phi = self.state.phi, tau = self.state.tau, mode = self.state.mode, regret = self.state.cumulative_regret },
        safe = (self.state.tau <= TAU_MAX)
    }
end

function AGI:get_status()
    local s = self.state
    return { status = self.status, phi = s.phi, tau = s.tau, mode = s.mode, regret = s.cumulative_regret }
end

return AGI
