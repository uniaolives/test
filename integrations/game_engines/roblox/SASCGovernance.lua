-- SASC Governance System for Roblox
-- Project Crux-86 Phase 3 - Article V Implementation

local SASC = {}
SASC.__index = SASC

-- Constants from Memory IDs
local CONSTANTS = {
    PHI_THRESHOLDS = {
        EXPLANATION = 0.65,
        PROPOSAL = 0.72,
        EMERGENCY = 0.78,
        HARD_FREEZE = 0.80
    },
    CORTISOL_DAMPING = 0.69, -- Dor do Boto protocol
    TMR_VARIANCE_THRESHOLD = 0.000032,
    SATOSHI_SEED = "0xbd36332890d15e2f360bb65775374b462b"
}

function SASC.new(options)
    local self = setmetatable({}, SASC)

    self.Phi = options.initialPhi or 0.65
    self.TMRInstances = {}
    self.Attestations = {}
    self.HardFreezeActive = false
    self.CortisolLevel = 0.0
    self.MeshNeuronEndpoint = options.meshEndpoint or "http://localhost:3030"

    -- Initialize TMR instances
    self:InitializeTMR()

    -- Start monitoring loop
    self.MonitoringTask = task.spawn(function()
        while true do
            self:MonitoringLoop()
            task.wait(0.1) -- 10Hz monitoring
        end
    end)

    print("[SASC] Governance system initialized with Φ =", self.Phi)
    return self
end

function SASC:InitializeTMR()
    -- Create 3 TMR instances (Pattern I40)
    for i = 1, 3 do
        self.TMRInstances[i] = {
            id = i,
            healthy = true,
            position = Vector3.new(0, 0, 0),
            gravity = Vector3.new(0, -196.2, 0), -- Roblox gravity (studs/s²)
            stateHash = "",
            lastUpdate = os.time()
        }
    end
    print("[TMR] Initialized 3 instances")
end

function SASC:ValidateAttestation(agentId, action, physicsState)
    -- Check Hard Freeze status
    if self.HardFreezeActive then
        return false, "HARD_FREEZE_ACTIVE"
    end

    -- Check Phi threshold
    if self.Phi < CONSTANTS.PHI_THRESHOLDS.PROPOSAL then
        return false, "INSUFFICIENT_PHI"
    end

    -- Update TMR state
    self:UpdateTMRState(physicsState)

    -- Validate TMR consensus
    local consensus, faultyInstance = self:CheckTMRConsensus()

    if not consensus then
        self:IsolateByzantineNode(faultyInstance, "STATE_DIVERGENCE")
        return false, "BYZANTINE_FAULT"
    end

    -- Check empathy stress levels
    if self.CortisolLevel > 0.3 then
        self:ApplyDorDoBotoProtocol()
    end

    -- Create EIP-712 style attestation
    local attestation = {
        agent = agentId,
        action = action,
        timestamp = os.time(),
        phi = self.Phi,
        stateHash = self:GenerateStateHash(physicsState),
        signature = self:SignAttestation(agentId, action)
    }

    table.insert(self.Attestations, attestation)

    -- Report to Mesh-Neuron
    self:ReportToMeshNeuron(attestation)

    return true, attestation
end

function SASC:UpdateTMRState(physicsState)
    for _, instance in ipairs(self.TMRInstances) do
        if instance.healthy then
            instance.position = physicsState.position or Vector3.new(0, 0, 0)
            instance.gravity = physicsState.gravity or Vector3.new(0, -196.2, 0)
            instance.lastUpdate = os.time()
            instance.stateHash = self:GenerateStateHash(physicsState)
        end
    end
end

function SASC:CheckTMRConsensus()
    -- Collect state hashes from healthy instances
    local hashes = {}
    for _, instance in ipairs(self.TMRInstances) do
        if instance.healthy and instance.stateHash ~= "" then
            hashes[instance.stateHash] = (hashes[instance.stateHash] or 0) + 1
        end
    end

    -- Find majority hash (2/3 consensus)
    local majorityHash, majorityCount = nil, 0
    for hash, count in pairs(hashes) do
        if count > majorityCount then
            majorityCount = count
            majorityHash = hash
        end
    end

    -- Check if we have consensus
    if majorityCount >= 2 then
        -- Check variance in positions
        local variance = self:CalculatePositionVariance()
        if variance > CONSTANTS.TMR_VARIANCE_THRESHOLD then
            -- Find faulty instance
            for _, instance in ipairs(self.TMRInstances) do
                if instance.healthy and instance.stateHash ~= majorityHash then
                    return false, instance.id
                end
            end
        end
        return true, nil
    else
        return false, 0 -- No consensus
    end
end

function SASC:CalculatePositionVariance()
    local positions = {}
    for _, instance in ipairs(self.TMRInstances) do
        if instance.healthy then
            table.insert(positions, instance.position.Y) -- Y axis for height
        end
    end

    if #positions < 2 then return 0 end

    -- Calculate mean
    local sum = 0
    for _, y in ipairs(positions) do
        sum = sum + y
    end
    local mean = sum / #positions

    -- Calculate variance
    local variance = 0
    for _, y in ipairs(positions) do
        variance = variance + (y - mean) * (y - mean)
    end
    variance = variance / #positions

    return variance
end

function SASC:TriggerHardFreeze(reason)
    if self.HardFreezeActive then return end

    self.HardFreezeActive = true
    print("[SASC HARD FREEZE] Reason:", reason, "Φ =", self.Phi)

    -- 1. Freeze all game scripts
    for _, player in ipairs(game:GetService("Players"):GetPlayers()) do
        if player.Character then
            local humanoid = player.Character:FindFirstChild("Humanoid")
            if humanoid then
                humanoid.PlatformStand = true
                humanoid.AutoRotate = false
            end
        end
    end

    -- 2. Seal state to KARNAK
    self:SealToKarnak({
        type = "hard_freeze",
        reason = reason,
        phi = self.Phi,
        timestamp = os.time(),
        satoshi_anchor = CONSTANTS.SATOSHI_SEED
    })

    -- 3. Notify Astraeus-1 Observer
    self:NotifyAstraeus({
        event = "HARD_FREEZE",
        severity = "CRITICAL",
        phi = self.Phi,
        reason = reason
    })

    -- 4. Broadcast to all clients
    game:GetService("ReplicatedStorage"):WaitForChild("SASCEvents"):FireAllClients(
        "HardFreeze",
        {reason = reason, phi = self.Phi}
    )
end

function SASC:ApplyDorDoBotoProtocol()
    local originalStress = self.CortisolLevel
    self.CortisolLevel = self.CortisolLevel * (1 - CONSTANTS.CORTISOL_DAMPING)

    print(string.format("[Dor do Boto] Stress reduced: %.2f → %.2f",
        originalStress, self.CortisolLevel))

    -- Reduce social complexity
    self:ReduceSocialComplexity(0.5)

    -- Increase cooperation rewards
    game:GetService("ReplicatedStorage"):WaitForChild("EconomyEvents"):FireAllClients(
        "AdjustCooperationBonus",
        2.0 -- Double cooperation rewards
    )
end

function SASC:ReduceSocialComplexity(factor)
    -- Reduce NPC density
    for _, npc in ipairs(workspace:GetChildren()) do
        if npc:IsA("Model") and npc.Name:match("NPC_") then
            local random = Random.new()
            if random:NextNumber() > factor then
                npc:Destroy()
            end
        end
    end

    -- Reduce particle effects
    for _, effect in ipairs(workspace:GetDescendants()) do
        if effect:IsA("ParticleEmitter") then
            effect.Rate = effect.Rate * factor
        end
    end
end

function SASC:MonitoringLoop()
    -- Update Phi based on system state
    local variance = self:CalculatePositionVariance()
    local newPhi = 1.0 - (variance * 1000)
    newPhi = math.max(0.0, math.min(1.0, newPhi))

    if math.abs(self.Phi - newPhi) > 0.01 then
        self.Phi = newPhi
        self:CheckPhiThresholds()
    end

    -- Check for timeout instances
    local currentTime = os.time()
    for _, instance in ipairs(self.TMRInstances) do
        if instance.healthy and (currentTime - instance.lastUpdate) > 5 then
            instance.healthy = false
            print("[TMR] Instance", instance.id, "timed out")
        end
    end

    -- Monitor social stress
    self:MonitorSocialStress()
end

function SASC:CheckPhiThresholds()
    if self.Phi >= CONSTANTS.PHI_THRESHOLDS.HARD_FREEZE then
        self:TriggerHardFreeze("PHI_CRITICAL_THRESHOLD")
    elseif self.Phi >= CONSTANTS.PHI_THRESHOLDS.EMERGENCY then
        print("[SASC] Emergency threshold reached, Φ =", self.Phi)
    elseif self.Phi >= CONSTANTS.PHI_THRESHOLDS.PROPOSAL then
        print("[SASC] Proposal threshold active, Φ =", self.Phi)
    end
end

function SASC:MonitorSocialStress()
    local players = game:GetService("Players"):GetPlayers()
    if #players == 0 then return end

    local totalStress = 0
    for _, player in ipairs(players) do
        local stress = self:CalculatePlayerStress(player)
        totalStress = totalStress + stress
    end

    self.CortisolLevel = totalStress / #players

    if self.CortisolLevel > 0.3 then
        self:ApplyDorDoBotoProtocol()
    end
end

function SASC:CalculatePlayerStress(player)
    local stress = 0

    -- Factor 1: Recent deaths
    local leaderstats = player:FindFirstChild("leaderstats")
    if leaderstats then
        local deaths = leaderstats:FindFirstChild("Deaths")
        if deaths and deaths.Value > 3 then
            stress = stress + 0.2
        end
    end

    -- Factor 2: Low health
    if player.Character then
        local humanoid = player.Character:FindFirstChild("Humanoid")
        if humanoid and humanoid.Health < 50 then
            stress = stress + 0.1
        end
    end

    -- Factor 3: Recent damage
    if player:GetAttribute("RecentDamage") then
        stress = stress + 0.15
    end

    return math.min(stress, 1.0)
end

function SASC:GenerateStateHash(physicsState)
    local stateString = string.format("%s|%s|%d",
        tostring(physicsState.position or Vector3.new(0,0,0)),
        tostring(physicsState.gravity or Vector3.new(0,-196.2,0)),
        os.time())

    -- Simple hash for demonstration
    local hash = 0
    for i = 1, #stateString do
        hash = hash + string.byte(stateString, i) * i
    end

    return string.format("%x", hash)
end

function SASC:SignAttestation(agentId, action)
    -- Simple signature simulation
    return string.format("sig_%s_%s_%d", agentId, action, os.time())
end

function SASC:ReportToMeshNeuron(data)
    -- HTTP request to Mesh-Neuron
    local success, result = pcall(function()
        local http = game:GetService("HttpService")
        return http:PostAsync(
            self.MeshNeuronEndpoint .. "/report",
            http:JSONEncode(data),
            Enum.HttpContentType.ApplicationJson
        )
    end)

    if not success then
        print("[Mesh-Neuron] Report failed:", result)
    end
end

function SASC:SealToKarnak(data)
    task.spawn(function()
        local success, result = pcall(function()
            local http = game:GetService("HttpService")
            return http:PostAsync(
                "http://localhost:9091/seal",
                http:JSONEncode(data),
                Enum.HttpContentType.ApplicationJson
            )
        end)

        if success then
            print("[KARNAK] State sealed:", data.type)
        else
            print("[KARNAK] Seal failed:", result)
        end
    end)
end

function SASC:NotifyAstraeus(data)
    print("[Astraeus-1] Event:", data.event, "Φ =", data.phi)
    -- In production, this would send to monitoring system
end

function SASC:IsolateByzantineNode(instanceId, reason)
    if self.TMRInstances[instanceId] then
        self.TMRInstances[instanceId].healthy = false
        print("[SASC] Isolated Byzantine node", instanceId, "-", reason)
    end
end

return SASC
