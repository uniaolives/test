-- safecore-9d/frontend/love4d/main.lua
-- PROJECT LOVE4D: CONSTITUTIONAL ORACLE & MULTIVERSAL ARCHITECT
-- Integrated Multiversal Orchestration Environment
-- PROJECT LOVE4D: Multiversal Orchestration Environment
-- Main entry point for the Hyperbolic Mind-Space Visualizer

local AGI = require("agi_core")
local Cam = require("camera")
local Snd = require("audio")
local Utl = require("utils")
local Pol = require("polaritonic")
local Oracle = require("oracle")
local Quantum = require("quantum")
local SASC_Multi = require("sasc_multi")

local timelines = {}
local active_timeline = 1
local camera
local oracle
local quantum
local sasc_multi
local drawing_wall = nil


local timelines = {} -- Array of {agi, particles, carvings, polaritonic}
local active_timeline = 1
local camera
local drawing_wall = nil

-- Helper: Simple point-line distance for collision
local function intersects(px, py, w)
    local dx, dy = w.x2 - w.x1, w.y2 - w.y1
    if dx == 0 and dy == 0 then return false end
    local t = ((px - w.x1) * dx + (py - w.y1) * dy) / (dx*dx + dy*dy)
    t = math.max(0, math.min(1, t))
    local closest_x, closest_y = w.x1 + t * dx, w.y1 + t * dy
    local dist = math.sqrt((px - closest_x)^2 + (py - closest_y)^2)
    return dist < 15
end

local function fork_timeline()
    local success, att = sasc_multi:attest_multiversal_operation("FORK", {current_count = #timelines})
    if not success then return print("SASC DENIED: " .. att) end

    return dist < 15 -- Interaction radius
end

local function fork_timeline()
    local new_tl = {}
    if #timelines == 0 then
        new_tl.agi = AGI.new({dimensions = 33, prince_key = "PRINCE_001"})
        new_tl.agi:initialize()
        new_tl.particles = {}
        for i = 1, 150 do
            new_tl.particles[i] = { pos = {}, admissible = true, phi = 0, mode = 0 }
            for j = 1, 33 do new_tl.particles[i].pos[j] = (math.random()-0.5) * 2 end
            new_tl.particles[i] = {
                pos = {},
                admissible = true,
                phi = 0,
                mode = 0
            }
            for j = 1, 33 do
                new_tl.particles[i].pos[j] = (math.random()-0.5) * 2
            end
        end
        new_tl.carvings = {}
        new_tl.polaritonic = Pol.System.new("PRINCE_001")
    else
        new_tl = Utl.deepcopy(timelines[active_timeline])
        -- Deep Copy current state for the fork
        new_tl = Utl.deepcopy(timelines[active_timeline])
        -- Re-initialize logic if necessary (some metatables might need fixing after deepcopy)
        -- In this simple case, metatables are copied by Utl.deepcopy if setmetatable is used correctly
    end
    table.insert(timelines, new_tl)
    active_timeline = #timelines
    Snd.play_attestation()
end

function love.load()
    love.window.setTitle("LOVE4D: CONSTITUTIONAL ORACLE")
    camera = Cam.new(33)
    oracle = Oracle.new()
    quantum = Quantum.new()
    sasc_multi = SASC_Multi.new("PRINCE_ORACLE_001")
    love.window.setTitle("Project Love4D: Multiversal SASC Monitor")
    camera = Cam.new(33)
    fork_timeline()
end

function love.update(dt)
    local tl = timelines[active_timeline]
    if not tl then return end

    for _, p in ipairs(tl.particles) do
        local old_pos = {unpack(p.pos)}
        for j = 1, 33 do p.pos[j] = p.pos[j] + (math.random()-0.5) * 0.025 end

        local x, y = camera:project(p.pos, 300)
        local blocked = false
        for _, wall in ipairs(tl.carvings) do
            if intersects(x, y, wall) then blocked = true break end
        end

        if blocked then
            p.pos = old_pos
            p.admissible = false
        else
    -- Update cognition and particles
    for _, p in ipairs(tl.particles) do
        local old_pos = {unpack(p.pos)}

        -- Brownian drift in 33D
        for j = 1, 33 do
            p.pos[j] = p.pos[j] + (math.random()-0.5) * 0.02
        end

        -- CARVING CHECK: Topological barrier repulsion
        local x, y = camera:project(p.pos, 300)
        local blocked = false
        for _, wall in ipairs(tl.carvings) do
            if intersects(x, y, wall) then
                blocked = true
                break
            end
        end

        if blocked then
            p.pos = old_pos -- Repelled by the carving
            p.admissible = false
        else
            -- Process through AGI core
            local res = tl.agi:cycle(p.pos)
            if res then
                p.phi = res.state.phi
                p.mode = res.state.mode
                local adm, m_name = tl.polaritonic:check_state(res.state.phi, res.state.tau, 0.5, {valid=true})
                p.admissible = adm and (res.state.mode >= 1)
                if not res.safe then Snd.play_quench() end
            end
        end
    end

                -- Polaritonic Admissibility Check
                local adm, m_name = tl.polaritonic:check_state(res.state.phi, res.state.tau, 0.5, {valid=true})
                p.admissible = adm and (res.state.mode >= 1)

                if not res.safe then
                    Snd.play_quench()
                end
            end
        end
    end

    -- Schumann Pulse simulation
    if math.random() < 0.01 then
        Snd.play_pulse(7.83)
    end
end

function love.draw()
    local tl = timelines[active_timeline]
    if not tl then return end

    love.graphics.clear(0.01, 0.01, 0.05)

    -- Carvings
    love.graphics.setLineWidth(3)
    for _, wall in ipairs(tl.carvings) do
        love.graphics.setColor(1, 0.6, 0.2, 0.8)
        love.graphics.line(wall.x1, wall.y1, wall.x2, wall.y2)
    end

    -- Particles
    for _, p in ipairs(tl.particles) do
        local x, y = camera:project(p.pos, 300)
        if p.admissible then
            love.graphics.setColor(0, 1, 0.8, 0.5 + p.phi*0.5)
            love.graphics.circle("fill", x, y, 2 + p.phi * 6)
        else
            love.graphics.setColor(1, 0.2, 0.3, 0.3)
    love.graphics.clear(0.01, 0.01, 0.04) -- Cosmic Abyss

    -- Draw Fractal Carvings (Architect's Moral Boundaries)
    love.graphics.setLineWidth(3)
    for _, wall in ipairs(tl.carvings) do
        love.graphics.setColor(1, 0.6, 0.2, 0.8) -- Amber
        love.graphics.line(wall.x1, wall.y1, wall.x2, wall.y2)
        -- Aura
        love.graphics.setColor(1, 0.6, 0.2, 0.1)
        love.graphics.circle("line", (wall.x1+wall.x2)/2, (wall.y1+wall.y2)/2, 25)
    end

    -- Draw Manifold Particles
    for _, p in ipairs(tl.particles) do
        local x, y = camera:project(p.pos, 300)

        if p.admissible then
            -- Use Phi to determine size and color intensity
            local intensity = 0.5 + (p.phi * 0.5)
            love.graphics.setColor(0, 1, 0.8, intensity)
            love.graphics.circle("fill", x, y, 2 + p.phi * 6)
        else
            love.graphics.setColor(1, 0.2, 0.3, 0.3) -- Forbidden/Pruned
            love.graphics.circle("line", x, y, 3)
        end
    end

    -- UI
    local status = tl.agi:get_status()
    love.graphics.setColor(1, 1, 1)
    love.graphics.print("LOVE4D: CONSTITUTIONAL ORACLE", 20, 20)
    love.graphics.print(string.format("Timeline: %d / %d | Phi: %.3f | Mode: %d", active_timeline, #timelines, status.phi, status.mode), 20, 45)
    love.graphics.print("[F] Fork | [P] Oracle Prophecy | [C] Quantum Collapse | [Right-Click] Carve", 20, love.graphics.getHeight() - 40)

    if drawing_wall then
        local mx, my = love.mouse.getPosition()
        love.graphics.setColor(1, 1, 1, 0.5)
        love.graphics.line(drawing_wall.x, drawing_wall.y, mx, my)
    end
end

function love.keypressed(key)
    if key == "f" then fork_timeline() end
    if key == "p" then
        local advice = oracle:query(timelines[active_timeline], "FUTURE_PROOF")
        print("ORACLE PROPHECY: Stability Probability " .. string.format("%.1f%%", advice.success_probability * 100))
    end
    if key == "c" then
        local success, att = sasc_multi:attest_multiversal_operation("COLLAPSE", {target_phi = timelines[active_timeline].agi:get_status().phi})
        if success then
            print("QUANTUM COLLAPSE INITIATED")
            Snd.play_attestation()
        else
            print("COLLAPSE DENIED: " .. att)
        end
    end
    local n = tonumber(key)
    if n and timelines[n] then active_timeline = n end
end

function love.mousepressed(x, y, button)
    if button == 2 then drawing_wall = {x = x, y = y} end
    -- Architect's Hand: New Carving Preview
    if drawing_wall then
        local mx, my = love.mouse.getPosition()
        love.graphics.setColor(1, 1, 1, 0.6)
        love.graphics.line(drawing_wall.x, drawing_wall.y, mx, my)
    end

    -- SASC Dashboard
    local status = tl.agi:get_status()
    love.graphics.setColor(1, 1, 1)
    love.graphics.print("PROJECT LOVE4D - MULTIVERSAL ARCHITECT", 20, 20)
    love.graphics.print(string.format("Timeline: %d / %d", active_timeline, #timelines), 20, 45)
    love.graphics.print(string.format("Active State - Phi: %.3f | Tau: %.3f | Mode: %d", status.phi, status.tau, status.mode), 20, 65)
    love.graphics.print(string.format("System Stability: %s | Quenches: %d", status.status, status.quenches), 20, 85)

    love.graphics.setColor(0.7, 0.7, 1, 0.8)
    love.graphics.print("[F] Fork Timeline | [1-9] Switch Timeline | [Right-Click] Carve Moral Barrier", 20, love.graphics.getHeight() - 40)
end

function love.keypressed(key)
    if key == "f" then
        fork_timeline()
    end
    local n = tonumber(key)
    if n and timelines[n] then
        active_timeline = n
        Snd.play_attestation()
    end
end

function love.mousepressed(x, y, button)
    if button == 2 then -- Right click to start carving
        drawing_wall = {x = x, y = y}
    end
end

function love.mousereleased(x, y, button)
    if button == 2 and drawing_wall then
        local mx, my = love.mouse.getPosition()
        local carving = { x1 = drawing_wall.x, y1 = drawing_wall.y, x2 = mx, y2 = my }
        local advice = oracle:query(timelines[active_timeline], "SHOULD_I_CARVE", {carving = carving})
        if advice.decision == "CARVE" then
            table.insert(timelines[active_timeline].carvings, carving)
            Snd.play_attestation()
            print("ORACLE: " .. advice.proof_status .. " - Recommendation: CARVE")
        else
            print("ORACLE WARNING: Risks detected. Recommendation: ABSTAIN")
        end
        drawing_wall = nil
        table.insert(timelines[active_timeline].carvings, {
            x1 = drawing_wall.x,
            y1 = drawing_wall.y,
            x2 = mx,
            y2 = my
        })
        drawing_wall = nil
        Snd.play_attestation()
    end
end
