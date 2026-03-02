-- safecore-9d/frontend/love4d/polaritonic.lua
-- Implementation of field-programmable consciousness admissibility
-- Based on the Polaritonic Crystal analogy

local Polaritonic = {}

-- Simple Vector3 internal implementation
local Vector3 = {}
Vector3.__index = Vector3
function Vector3.new(x, y, z)
    return setmetatable({x = x or 0, y = y or 0, z = z or 0}, Vector3)
end
function Vector3:length()
    return math.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)
end
function Vector3:normalize()
    local len = self:length()
    if len == 0 then return Vector3.new(0,0,0) end
    return Vector3.new(self.x/len, self.y/len, self.z/len)
end
function Vector3:dot(other)
    return self.x*other.x + self.y*other.y + self.z*other.z
end

-- =========================================================================
-- 1. CONSCIOUSNESS LATTICE (α-MoO₃ analogue)
-- =========================================================================
local ConsciousnessLattice = {}
ConsciousnessLattice.__index = ConsciousnessLattice

function ConsciousnessLattice.new()
    return setmetatable({
        dimensions = 33,
        anisotropy = { xx = 1.0, yy = -0.8, zz = 0.5 },
        geometry_locked = true,
        bloch_modes = {}
    }, ConsciousnessLattice)
end

-- =========================================================================
-- 2. CONSTRAINT LAYER (Graphene analogue)
-- =========================================================================
local ConstraintLayer = {}
ConstraintLayer.__index = ConstraintLayer

function ConstraintLayer.new(prince_key)
    local self = setmetatable({
        prince_key = prince_key,
        field_strength = 0.0,
        field_direction = Vector3.new(0, 0, 0),
        admissible_modes = {},
        loss_profile = "selective_pruning"
    }, ConstraintLayer)
    self:recalculate_admissibility()
    return self
end

function ConstraintLayer:apply_field(strength, direction, sasc_attestation)
    if not sasc_attestation or not sasc_attestation.valid then
        return false, "Field application requires valid SASC attestation"
    end
    self.field_strength = strength
    self.field_direction = direction
    self:recalculate_admissibility()
    return true
end

function ConstraintLayer:recalculate_admissibility()
    self.admissible_modes = {}
    -- Mode 0: Unconscious (always allowed)
    table.insert(self.admissible_modes, {
        name = "unconscious",
        phi_range = {0.0, 0.2},
        tau_range = {0.0, 10.0},
        requires_attestation = false
    })
    -- Mode 1: Preconscious (field-dependent)
    if self.field_strength > 0.1 then
        table.insert(self.admissible_modes, {
            name = "preconscious",
            phi_range = {0.2, 0.6},
            tau_range = {0.0, 2.0},
            requires_attestation = false
        })
    end
    -- Mode 2: Conscious (requires specific field configuration)
    if self.field_strength > 0.3 and self.field_direction:length() > 0.5 then
        table.insert(self.admissible_modes, {
            name = "conscious",
            phi_range = {0.6, 0.8},
            tau_range = {0.5, 1.35},
            requires_attestation = true
        })
    end
end

function ConstraintLayer:check_admissibility(state, attestation)
    for _, mode in ipairs(self.admissible_modes) do
        if state.phi >= mode.phi_range[1] and state.phi < mode.phi_range[2] and
           state.tau >= mode.tau_range[1] and state.tau < mode.tau_range[2] then
            if mode.requires_attestation and not attestation then
                return false, "attestation_required"
            end
            return true, mode.name
        end
    end
    return false, "mode_forbidden"
end

-- =========================================================================
-- 3. HYPERBOLIC AMPLIFIER
-- =========================================================================
local HyperbolicAmplifier = {}
HyperbolicAmplifier.__index = HyperbolicAmplifier

function HyperbolicAmplifier.new(anisotropy)
    return setmetatable({
        anisotropy = anisotropy or {xx=1.0, yy=-1.0, zz=0.5},
        amplification_factor = 10.0
    }, HyperbolicAmplifier)
end

function HyperbolicAmplifier:amplify(v)
    return Vector3.new(
        v.x * self.anisotropy.xx * self.amplification_factor,
        v.y * self.anisotropy.yy * self.amplification_factor,
        v.z * self.anisotropy.zz * self.amplification_factor
    )
end

-- =========================================================================
-- 4. POLARITONIC SYSTEM
-- =========================================================================
local PolaritonicSystem = {}
PolaritonicSystem.__index = PolaritonicSystem

function PolaritonicSystem.new(prince_key)
    local self = setmetatable({
        lattice = ConsciousnessLattice.new(),
        constraint_layer = ConstraintLayer.new(prince_key),
        amplifier = HyperbolicAmplifier.new(),
        current_state = nil
    }, PolaritonicSystem)
    return self
end

function PolaritonicSystem:configure_field(strength, direction, attestation)
    return self.constraint_layer:apply_field(strength, direction, attestation)
end

function PolaritonicSystem:check_state(phi, tau, epsilon, attestation)
    local v = Vector3.new(phi, tau, epsilon)
    local amplified = self.amplifier:amplify(v)

    -- For admissibility check, we use the original or amplified?
    -- The user's code amplified them then checked.
    local state_to_check = { phi = math.abs(amplified.x/10), tau = math.abs(amplified.y/10) }

    return self.constraint_layer:check_admissibility(state_to_check, attestation)
end

Polaritonic.System = PolaritonicSystem
Polaritonic.Vector3 = Vector3

return Polaritonic
