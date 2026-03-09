-- orb_script.lua

Orb = {}
Orb.__index = Orb

function Orb:new(lambda, freq)
    local self = setmetatable({}, Orb)
    self.stability = lambda
    self.frequency = freq
    return self
end

function Orb:transmit(handover)
    if self.stability > 0.5 then
        return true, "Transmitted"
    else
        return false, "Collapsed"
    end
end

return Orb
