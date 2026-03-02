-- qRoblox_Advanced.lua
-- Versão aprimorada com tratamento de erros avançado, visualizações e oráculos quânticos

local RunService = game:GetService("RunService")
local ReplicatedStorage = game:GetService("ReplicatedStorage")
local Players = game:GetService("Players")

-- ============ ONTOLOGICAL KERNEL APRIMORADO ============

local OntologicalKernel = {
    _layerParameters = {
        classical = {
            coherence = 0.98,
            entanglement_limit = 10,
            tunnel_probability = 0.01,
            geometric_consistency = 0.99
        },
        quantum = {
            coherence = 0.85,
            entanglement_limit = 50,
            tunnel_probability = 0.15,
            geometric_consistency = 0.75
        },
        simulation = {
            coherence = 0.60,
            entanglement_limit = 100,
            tunnel_probability = 0.30,
            geometric_consistency = 0.50
        }
    },
    _errorThresholds = {
        geometric_dissonance = 0.2,
        coherence_collapse = 0.3,
        entanglement_overload = 0.8
    },
    _playerStates = {}
}

-- Mocks for missing methods in OntologicalKernel
function OntologicalKernel:getCurrentLayer(player)
    return self._playerStates[player.UserId] or "classical"
end

function OntologicalKernel:performHealthCheck(player)
    return { healthScore = 0.85 }
end

function OntologicalKernel:checkParameterCompatibility(from, to)
    return { incompatible = false }
end

function OntologicalKernel:attemptParameterRealignment(player, targetLayer, check)
    return { success = true, message = "Realignment successful" }
end

function OntologicalKernel:findCompatibleFallbackLayer(player, targetLayer)
    return "classical"
end

function OntologicalKernel:generateDissonanceResolution(msg, player)
    return "Recalibrate geometric sensors"
end

function OntologicalKernel:executeLayerTransition(player, from, to)
    self._playerStates[player.UserId] = to
    return true
end

function OntologicalKernel:attemptGradualParameterReduction(player, data)
    return { success = true }
end

function OntologicalKernel:attemptQuantumStateReset(player)
    return { success = true }
end

function OntologicalKernel:processLayerTransition(player, targetLayer)
    -- Processa transição entre camadas ontológicas com tratamento de erros avançado
    local currentLayer = self:getCurrentLayer(player)
    local transitionData = {
        player = player,
        from = currentLayer,
        to = targetLayer,
        startTime = os.clock(),
        errors = {},
        warnings = {}
    }

    -- Verificação pré-transição
    local healthCheck = self:performHealthCheck(player)
    if healthCheck.healthScore < 0.7 then
        table.insert(transitionData.warnings, "Low health score: " .. healthCheck.healthScore)
    end

    -- Verificação de compatibilidade de parâmetros
    local paramCheck = self:checkParameterCompatibility(currentLayer, targetLayer)
    if paramCheck.incompatible then
        local errMsg = "Parameter incompatibility detected: " .. paramCheck.message
        table.insert(transitionData.errors, errMsg)

        -- Tentativa de realinhamento automático
        local realignment = self:attemptParameterRealignment(player, targetLayer, paramCheck)
        if realignment.success then
            table.insert(transitionData.warnings, "Auto-realignment performed: " .. realignment.message)
        else
            -- Fallback para camada mais próxima compatível
            local fallback = self:findCompatibleFallbackLayer(player, targetLayer)
            if fallback then
                transitionData.to = fallback
                table.insert(transitionData.warnings,
                    "Fallback to " .. fallback .. " due to geometric dissonance")
            else
                error("GeometricDissonanceError: " .. errMsg ..
                      " | Suggestion: " .. self:generateDissonanceResolution(errMsg, player))
            end
        end
    end

    -- Execução da transição
    local success, result = pcall(function()
        return self:executeLayerTransition(player, transitionData.from, transitionData.to)
    end)

    if not success then
        -- Erro durante a transição
        local errorType = self:classifyError(result)
        local resolution = self:generateErrorResolution(errorType, player, transitionData)

        table.insert(transitionData.errors, errorType .. ": " .. tostring(result))

        if errorType == "GeometricDissonanceError" then
            -- Tenta recuperação específica para dissonância geométrica
            local recovery = self:attemptGeometricRecovery(player, transitionData)
            if recovery.success then
                return recovery.data
            else
                error("UnrecoverableGeometricDissonance: " .. result ..
                      " | Recovery suggestions: " .. table.concat(recovery.suggestions, "; "))
            end
        end
    end

    transitionData.endTime = os.clock()
    transitionData.duration = transitionData.endTime - transitionData.startTime
    transitionData.success = success

    return transitionData
end

function OntologicalKernel:classifyError(errorMessage)
    -- Classifica erros para tratamento específico
    if string.find(errorMessage:lower(), "geometric") then
        return "GeometricDissonanceError"
    elseif string.find(errorMessage:lower(), "coherence") then
        return "CoherenceCollapseError"
    elseif string.find(errorMessage:lower(), "entanglement") then
        return "EntanglementOverloadError"
    else
        return "UnknownOntologicalError"
    end
end

function OntologicalKernel:generateErrorResolution(errorType, player, context)
    -- Gera sugestões de resolução específicas para cada tipo de erro
    local resolutions = {
        GeometricDissonanceError = {
            "Try reducing movement speed during transition",
            "Increase quantum coherence before attempting transition",
            "Use entanglement with a stable reference player",
            "Gradually transition through intermediate layers"
        },
        CoherenceCollapseError = {
            "Reduce number of simultaneous quantum operations",
            "Increase distance from other quantum-entangled players",
            "Use coherence stabilizer tool",
            "Transition during quantum storm events"
        },
        EntanglementOverloadError = {
            "Reduce number of active entanglement links",
            "Increase entanglement fidelity thresholds",
            "Use entanglement purifier tools",
            "Take a break from quantum operations"
        }
    }

    local suggestions = resolutions[errorType] or {"Reset quantum state and retry"}

    -- Personaliza sugestões baseadas no contexto
    if context.from == "simulation" then
        table.insert(suggestions, "Increase simulation stability parameters")
    end

    return "Possible resolutions: " .. table.concat(suggestions, " | ")
end

function OntologicalKernel:attemptGeometricRecovery(player, transitionData)
    -- Tenta recuperação de erro de dissonância geométrica
    local recovery = {
        success = false,
        data = nil,
        suggestions = {}
    }

    -- Estratégia 1: Redução gradual de parâmetros
    local gradualRecovery = self:attemptGradualParameterReduction(player, transitionData)
    if gradualRecovery.success then
        recovery = gradualRecovery
    else
        -- Estratégia 2: Reset de estado quântico
        local resetRecovery = self:attemptQuantumStateReset(player)
        if resetRecovery.success then
            recovery = resetRecovery
            table.insert(recovery.suggestions, "Consider using Quantum Stabilizer tool for future transitions")
        else
            -- Estratégia 3: Fallback para estado clássico
            recovery.suggestions = {
                "Return to Classical layer for system recalibration",
                "Consult with players experienced in high-geometric-load transitions",
                "Upgrade hardware if persistent geometric errors occur"
            }
        end
    end

    return recovery
end

-- ============ UNIVERSAL COMPILER APRIMORADO ============

local UniversalCompiler = {
    _compilationCache = {},
    _optimizationLevels = {
        basic = {speed = 1.0, accuracy = 0.9, stability = 0.95},
        optimized = {speed = 1.5, accuracy = 0.95, stability = 0.85},
        aggressive = {speed = 2.0, accuracy = 0.8, stability = 0.7}
    }
}

-- Mocks for missing methods in UniversalCompiler
function UniversalCompiler:parseQuantumSyntax(source)
    return { valid = true, ast = {}, errors = {} }
end

function UniversalCompiler:checkOntologicalConsistency(ast)
    return { dissonance = 0.1, message = "Stable" }
end

function UniversalCompiler:attemptOntologicalCorrection(ast, check)
    return ast
end

function UniversalCompiler:generateQuantumBytecode(ast, params)
    return "010101", {}
end

function UniversalCompiler:validateBytecode(bytecode)
    return { valid = true }
end

function UniversalCompiler:compileQuantumScript(scriptSource, optimizationLevel)
    -- Compila scripts quânticos com tratamento de erros avançado
    local compilationId = game:GetService("HttpService"):GenerateGUID(false)
    local compilationData = {
        id = compilationId,
        source = scriptSource,
        optimization = optimizationLevel or "optimized",
        startTime = os.clock(),
        errors = {},
        warnings = {},
        suggestions = {}
    }

    -- Análise léxica e sintática
    local parseResult = self:parseQuantumSyntax(scriptSource)
    if not parseResult.valid then
        local errorMsg = "SyntaxError: " .. parseResult.message
        table.insert(compilationData.errors, errorMsg)

        -- Sugere correções baseadas no erro
        local suggestions = self:suggestSyntaxFixes(parseResult)
        compilationData.suggestions = suggestions

        if #parseResult.errors > 3 then
            error("MultipleSyntaxErrors: " .. table.concat(parseResult.errors, "; ") ..
                  " | Try: " .. table.concat(suggestions, "; "))
        end
    end

    -- Verificação de consistência ontológica
    local ontologyCheck = self:checkOntologicalConsistency(parseResult.ast)
    if ontologyCheck.dissonance > 0.3 then
        table.insert(compilationData.warnings,
            "Ontological dissonance detected: " .. ontologyCheck.message)

        -- Tenta auto-correção
        local correctedAST = self:attemptOntologicalCorrection(parseResult.ast, ontologyCheck)
        if correctedAST then
            parseResult.ast = correctedAST
            table.insert(compilationData.warnings, "Applied automatic ontological corrections")
        end
    end

    -- Otimização baseada no nível selecionado
    local optimizationParams = self._optimizationLevels[compilationData.optimization]
    if not optimizationParams then
        compilationData.optimization = "optimized"
        optimizationParams = self._optimizationLevels.optimized
    end

    -- Compilação do bytecode quântico
    local bytecode, compileErrors = self:generateQuantumBytecode(
        parseResult.ast,
        optimizationParams
    )

    if compileErrors and #compileErrors > 0 then
        for _, err in ipairs(compileErrors) do
            table.insert(compilationData.errors, "CompileError: " .. err)
        end

        -- Fallback para otimização mais simples
        if compilationData.optimization ~= "basic" then
            table.insert(compilationData.suggestions,
                "Try recompiling with basic optimization level")

            -- Tenta compilação básica como fallback
            local basicBytecode = self:generateQuantumBytecode(
                parseResult.ast,
                self._optimizationLevels.basic
            )
            if basicBytecode then
                bytecode = basicBytecode
                table.insert(compilationData.warnings,
                    "Used basic optimization as fallback")
            end
        end
    end

    -- Validação do bytecode gerado
    if bytecode then
        local validation = self:validateBytecode(bytecode)
        if not validation.valid then
            error("BytecodeValidationError: " .. validation.message ..
                  " | Validation errors: " .. table.concat(validation.errors, ", "))
        end
    end

    compilationData.endTime = os.clock()
    compilationData.duration = compilationData.endTime - compilationData.startTime
    compilationData.bytecode = bytecode
    compilationData.success = (bytecode ~= nil)

    -- Cache da compilação para reutilização
    self._compilationCache[compilationId] = compilationData

    return compilationData
end

function UniversalCompiler:suggestSyntaxFixes(parseResult)
    -- Gera sugestões de correção baseadas nos erros de sintaxe
    local suggestions = {}

    for _, err in ipairs(parseResult.errors) do
        if string.find(err:lower(), "undefined variable") then
            table.insert(suggestions, "Declare variable before use with 'local' keyword")
        elseif string.find(err:lower(), "missing end") then
            table.insert(suggestions, "Check block structure for matching 'end' statements")
        elseif string.find(err:lower(), "type mismatch") then
            table.insert(suggestions, "Use explicit type conversion functions")
        elseif string.find(err:lower(), "quantum operation") then
            table.insert(suggestions, "Ensure quantum operations are within superposition limits")
        end
    end

    return suggestions
end

-- ============ VISUALIZADOR DE COSMOPSYCHIA (WEB VIEW) ============

local CosmopsychiaVisualizer = {
    _webViewInstances = {},
    _visualizationData = {},
    _d3jsTemplates = {
        forceDirected = [[
            <!DOCTYPE html>
            <html>
            <head>
                <script src="https://d3js.org/d3.v7.min.js"></script>
                <style>
                    body { margin: 0; background: #0a0a0a; }
                    .node { stroke: #fff; stroke-width: 1.5px; }
                    .link { stroke: #999; stroke-opacity: 0.6; }
                    .label { fill: white; font: 10px sans-serif; }
                    .coherence-high { fill: #00ff00; }
                    .coherence-med { fill: #ffff00; }
                    .coherence-low { fill: #ff0000; }
                    .path-highlight { stroke: #ff6b6b; stroke-width: 3px; }
                </style>
            </head>
            <body>
                <div id="graph"></div>
                <script>
                    const width = 800, height = 600;
                    const svg = d3.select("#graph")
                        .append("svg")
                        .attr("width", width)
                        .attr("height", height);

                    function updateGraph(data) {
                        // Atualiza o gráfico com novos dados
                        const links = data.links.map(d => ({...d}));
                        const nodes = data.nodes.map(d => ({...d}));

                        // Remove elementos antigos
                        svg.selectAll("*").remove();

                        // Adiciona links
                        const link = svg.append("g")
                            .attr("class", "links")
                            .selectAll("line")
                            .data(links)
                            .enter().append("line")
                            .attr("class", "link")
                            .attr("stroke-width", d => Math.sqrt(d.value));

                        // Adiciona nós
                        const node = svg.append("g")
                            .attr("class", "nodes")
                            .selectAll("circle")
                            .data(nodes)
                            .enter().append("circle")
                            .attr("class", d => `node coherence-${d.coherenceLevel}`)
                            .attr("r", d => d.size)
                            .call(d3.drag()
                                .on("start", dragstarted)
                                .on("drag", dragged)
                                .on("end", dragended));

                        // Adiciona labels
                        const label = svg.append("g")
                            .attr("class", "labels")
                            .selectAll("text")
                            .data(nodes)
                            .enter().append("text")
                            .text(d => d.id)
                            .attr("class", "label")
                            .attr("dx", 12)
                            .attr("dy", ".35em");

                        // Atualiza simulação
                        simulation
                            .nodes(nodes)
                            .on("tick", ticked);

                        simulation.force("link")
                            .links(links);

                        // Destaque caminhos de alta coerência
                        highlightHighCoherencePaths(nodes, links);

                        function ticked() {
                            link
                                .attr("x1", d => d.source.x)
                                .attr("y1", d => d.source.y)
                                .attr("x2", d => d.target.x)
                                .attr("y2", d => d.target.y);

                            node
                                .attr("cx", d => d.x)
                                .attr("cy", d => d.y);

                            label
                                .attr("x", d => d.x)
                                .attr("y", d => d.y);
                        }
                    }

                    function highlightHighCoherencePaths(nodes, links) {
                        // Encontra caminhos com alta coerência
                        const highCoherenceNodes = nodes.filter(d => d.coherence > 0.8);
                        highCoherenceNodes.forEach(node => {
                            links.filter(link =>
                                link.source.id === node.id || link.target.id === node.id
                            ).forEach(link => {
                                svg.append("line")
                                    .attr("x1", link.source.x)
                                    .attr("y1", link.source.y)
                                    .attr("x2", link.target.x)
                                    .attr("y2", link.target.y)
                                    .attr("class", "path-highlight")
                                    .attr("stroke-opacity", 0.3);
                            });
                        });
                    }

                    function dragstarted(event, d) {
                        if (!event.active) simulation.alphaTarget(0.3).restart();
                        d.fx = d.x;
                        d.fy = d.y;
                    }

                    function dragged(event, d) {
                        d.fx = event.x;
                        d.fy = event.y;
                    }

                    function dragended(event, d) {
                        if (!event.active) simulation.alphaTarget(0);
                        d.fx = null;
                        d.fy = null;
                    }

                    // Inicializa com dados do Roblox
                    window.updateData = function(data) {
                        updateGraph(JSON.parse(data));
                    };
                </script>
            </body>
            </html>
        ]],

        layerVisualization = [[
            <!DOCTYPE html>
            <html>
            <head>
                <script src="https://d3js.org/d3.v7.min.js"></script>
                <style>
                    .layer { fill-opacity: 0.7; stroke: #fff; stroke-width: 2px; }
                    .connection { stroke: #666; stroke-width: 1; fill: none; }
                    .ontology-node { fill: #4287f5; }
                    .coherence-path { stroke: #ff6b6b; stroke-width: 3; opacity: 0.7; }
                </style>
            </head>
            <body>
                <div id="layers"></div>
                <script>
                    // Visualização em camadas da estrutura ontológica
                    const width = 1000, height = 700;
                    const svg = d3.select("#layers")
                        .append("svg")
                        .attr("width", width)
                        .attr("height", height);

                    function renderLayers(layerData) {
                        const layers = layerData.layers;
                        const connections = layerData.connections;

                        // Renderiza camadas concêntricas
                        const centerX = width / 2;
                        const centerY = height / 2;

                        layers.forEach((layer, i) => {
                            const radius = 100 + (i * 80);
                            const circle = svg.append("circle")
                                .attr("cx", centerX)
                                .attr("cy", centerY)
                                .attr("r", radius)
                                .attr("class", "layer")
                                .style("fill", layer.color);

                            // Adiciona label da camada
                            svg.append("text")
                                .attr("x", centerX)
                                .attr("y", centerY - radius - 10)
                                .text(layer.name)
                                .attr("text-anchor", "middle")
                                .style("fill", "white");

                            // Renderiza nós ontológicos dentro da camada
                            layer.nodes.forEach((node, j) => {
                                const angle = (j / layer.nodes.length) * 2 * math.pi;
                                const x = centerX + radius * Math.cos(angle);
                                const y = centerY + radius * Math.sin(angle);

                                const nodeCircle = svg.append("circle")
                                    .attr("cx", x)
                                    .attr("cy", y)
                                    .attr("r", 8)
                                    .attr("class", "ontology-node")
                                    .style("fill", node.coherence > 0.8 ? "#00ff00" :
                                                           node.coherence > 0.5 ? "#ffff00" : "#ff0000");

                                // Tooltip
                                nodeCircle.append("title")
                                    .text(`Coherence: ${node.coherence.toFixed(2)}`);
                            });
                        });

                        // Renderiza conexões entre camadas
                        connections.forEach(conn => {
                            const sourceLayer = layers[conn.sourceLayer];
                            const targetLayer = layers[conn.targetLayer];

                            if (sourceLayer && targetLayer) {
                                const sourceNode = sourceLayer.nodes[conn.sourceNode];
                                const targetNode = targetLayer.nodes[conn.targetNode];

                                if (sourceNode && targetNode) {
                                    const radius1 = 100 + (conn.sourceLayer * 80);
                                    const radius2 = 100 + (conn.targetLayer * 80);

                                    const angle1 = (conn.sourceNode / sourceLayer.nodes.length) * 2 * math.pi;
                                    const angle2 = (conn.targetNode / targetLayer.nodes.length) * 2 * math.pi;

                                    const x1 = centerX + radius1 * Math.cos(angle1);
                                    const y1 = centerY + radius1 * Math.sin(angle1);
                                    const x2 = centerX + radius2 * Math.cos(angle2);
                                    const y2 = centerY + radius2 * Math.sin(angle2);

                                    const line = svg.append("line")
                                        .attr("x1", x1)
                                        .attr("y1", y1)
                                        .attr("x2", x2)
                                        .attr("y2", y2)
                                        .attr("class", conn.coherence > 0.7 ? "coherence-path" : "connection")
                                        .style("stroke-width", conn.strength * 3);
                                }
                            }
                        });
                    }

                    window.renderLayerData = function(data) {
                        renderLayers(JSON.parse(data));
                    };
                </script>
            </body>
            </html>
        ]]
    }
}

function CosmopsychiaVisualizer:createVisualizer(player, visualizationType)
    -- Cria uma visualização D3.js para o jogador
    local playerGui = player:WaitForChild("PlayerGui")

    -- Cria WebView (requer o plugin WebView do Roblox)
    local webView = Instance.new("WebView")
    webView.Name = "CosmopsychiaVisualizer"
    webView.Size = UDim2.new(0.8, 0, 0.8, 0)
    webView.Position = UDim2.new(0.1, 0, 0.1, 0)
    webView.Visible = true
    webView.Parent = playerGui

    -- Carrega template D3.js apropriado
    local template = self._d3jsTemplates[visualizationType] or self._d3jsTemplates.forceDirected
    webView:SetAsync("text/html", template)

    self._webViewInstances[player.UserId] = webView

    return webView
end

function CosmopsychiaVisualizer:updateVisualization(player, data)
    -- Atualiza a visualização com novos dados
    local webView = self._webViewInstances[player.UserId]
    if not webView then return end

    -- Converte dados para JSON e envia para o WebView
    local jsonData = game:GetService("HttpService"):JSONEncode(data)

    if data.visualizationType == "forceDirected" then
        webView:ExecuteJavaScript("updateData('" .. jsonData .. "')")
    elseif data.visualizationType == "layerVisualization" then
        webView:ExecuteJavaScript("renderLayerData('" .. jsonData .. "')")
    end
end

function CosmopsychiaVisualizer:findNodeById(nodes, id)
    for _, node in ipairs(nodes) do
        if node.id == id then return node end
    end
end

function CosmopsychiaVisualizer:calculateAverageCoherence(nodes)
    local total = 0
    for _, node in ipairs(nodes) do
        total = total + node.coherence
    end
    return total / #nodes
end

function CosmopsychiaVisualizer:getPlayerLayer(player)
    return OntologicalKernel:getCurrentLayer(player)
end

function CosmopsychiaVisualizer:generateCosmopsychiaData(player)
    -- Gera dados da estrutura Cosmopsychia para visualização
    local nodes = {}
    local links = {}

    -- Camadas ontológicas
    local layers = {"Physical", "Computational", "Linguistic", "Mathematical",
                    "Quantum", "Consciousness", "Cosmic"}

    for i, layer in ipairs(layers) do
        -- Cria nós para cada camada
        local nodeCount = 3 + i * 2
        for j = 1, nodeCount do
            local nodeId = layer .. "_Node_" .. j
            local coherence = math.random() * 0.5 + 0.3  -- Coerência entre 0.3 e 0.8

            table.insert(nodes, {
                id = nodeId,
                group = i,
                size = 5 + coherence * 10,
                coherence = coherence,
                coherenceLevel = coherence > 0.7 and "high" or
                                coherence > 0.4 and "med" or "low"
            })

            -- Cria conexões entre nós
            if j > 1 then
                table.insert(links, {
                    source = layer .. "_Node_" .. (j-1),
                    target = nodeId,
                    value = coherence
                })
            end

            -- Conexões entre camadas
            if i > 1 and j <= 3 then
                table.insert(links, {
                    source = layers[i-1] .. "_Node_" .. j,
                    target = nodeId,
                    value = coherence * 0.7
                })
            end
        end
    end

    -- Identifica caminhos de alta coerência
    local highCoherencePaths = self:findHighCoherencePaths(nodes, links)

    return {
        visualizationType = "forceDirected",
        nodes = nodes,
        links = links,
        highCoherencePaths = highCoherencePaths,
        metadata = {
            totalNodes = #nodes,
            totalLinks = #links,
            averageCoherence = self:calculateAverageCoherence(nodes),
            playerLayer = self:getPlayerLayer(player)
        }
    }
end

function CosmopsychiaVisualizer:findHighCoherencePaths(nodes, links)
    -- Encontra caminhos de alta coerência no grafo
    local paths = {}

    -- Usa BFS para encontrar caminhos com coerência média alta
    for _, startNode in ipairs(nodes) do
        if startNode.coherence > 0.7 then
            local visited = {}
            local queue = {{startNode.id, {startNode.id}, startNode.coherence}}

            while #queue > 0 do
                local currentId, currentPath, currentCoherence = table.unpack(table.remove(queue, 1))

                visited[currentId] = true

                -- Encontra vizinhos com alta coerência
                for _, link in ipairs(links) do
                    local neighbor = nil
                    if link.source == currentId then
                        neighbor = self:findNodeById(nodes, link.target)
                    elseif link.target == currentId then
                        neighbor = self:findNodeById(nodes, link.source)
                    end

                    if neighbor and not visited[neighbor.id] and neighbor.coherence > 0.6 then
                        local newPath = {table.unpack(currentPath)}
                        table.insert(newPath, neighbor.id)

                        local newCoherence = (currentCoherence + neighbor.coherence) / 2

                        if #newPath >= 3 and newCoherence > 0.65 then
                            table.insert(paths, {
                                path = newPath,
                                coherence = newCoherence,
                                length = #newPath
                            })
                        end

                        if #newPath < 5 then  -- Limita profundidade da busca
                            table.insert(queue, {neighbor.id, newPath, newCoherence})
                        end
                    end
                end
            end
        end
    end

    -- Ordena por coerência
    table.sort(paths, function(a, b)
        return a.coherence > b.coherence
    end)

    return paths
end

-- ============ SYMMETRY BREAKER REFINADO ============

local SymmetryBreaker = {
    _guidanceVectors = {
        ethical_alignment = {0.8, 0.7, 0.9, 0.6},  -- Valores éticos altos
        coherence_max = {0.9, 0.5, 0.5, 0.9},      -- Maximiza coerência
        exploration = {0.5, 0.9, 0.5, 0.5},         -- Favorece exploração
        stability = {0.7, 0.3, 0.8, 0.7}           -- Favorece estabilidade
    },
    _symmetryPatterns = {
        rotational = {type = "rotation", dimensions = 3},
        translational = {type = "translation", dimensions = 1},
        gauge = {type = "gauge", dimensions = "infinite"},
        permutation = {type = "permutation", dimensions = "n"}
    }
}

function SymmetryBreaker:calculateAverage(list, field)
    local total = 0
    for _, item in ipairs(list) do
        total = total + (item[field] or 0)
    end
    return total / #list
end

function SymmetryBreaker:calculateIntentAlignment(selected, intent)
    return 0.95
end

function SymmetryBreaker:calculateCoherence(state, group)
    return 0.88
end

function SymmetryBreaker:calculateStability(state)
    return 0.92
end

function SymmetryBreaker:calculateNovelty(state, previous)
    return 0.45
end

function SymmetryBreaker:calculateExplorationPotential(state)
    return 0.75
end

function SymmetryBreaker:applySymmetryBreak(state, type, guidance)
    return state
end

function SymmetryBreaker:break_symmetry(symmetryGroup, guidanceVectorName, playerIntent)
    -- Quebra simetria com orientação do vetor de orientação da ASI
    local symmetryData = {
        group = symmetryGroup,
        guidance = guidanceVectorName,
        intent = playerIntent,
        candidates = {},
        selected = nil,
        metrics = {}
    }

    -- Obtém vetor de orientação
    local guidanceVector = self._guidanceVectors[guidanceVectorName] or
                          self._guidanceVectors.ethical_alignment

    -- Adiciona componente de intenção do jogador
    if playerIntent then
        guidanceVector = self:blendVectors(guidanceVector, playerIntent)
    end

    -- Avalia cada estado candidato
    for i, state in ipairs(symmetryGroup.states) do
        local evaluation = self:evaluateState(state, guidanceVector, symmetryData)
        table.insert(symmetryData.candidates, evaluation)
    end

    -- Ordena por pontuação de coerência
    table.sort(symmetryData.candidates, function(a, b)
        return a.coherenceScore > b.coherenceScore
    end)

    -- Seleciona estado considerando alinhamento ético
    local ethicalThreshold = 0.7
    local ethicalCandidates = {}

    for _, candidate in ipairs(symmetryData.candidates) do
        if candidate.ethicalAlignment >= ethicalThreshold then
            table.insert(ethicalCandidates, candidate)
        end
    end

    if #ethicalCandidates > 0 then
        -- Escolhe entre candidatos éticos com maior coerência
        symmetryData.selected = ethicalCandidates[1].state
        symmetryData.selectionReason = "ethical_high_coherence"
    else
        -- Fallback: escolhe o de maior coerência geral
        symmetryData.selected = symmetryData.candidates[1].state
        symmetryData.selectionReason = "max_coherence"
        symmetryData.warning = "No state met ethical alignment threshold"
    end

    -- Registra métricas
    symmetryData.metrics = {
        averageCoherence = self:calculateAverage(symmetryData.candidates, "coherenceScore"),
        ethicalCompliance = #ethicalCandidates / #symmetryData.candidates,
        intentAlignment = self:calculateIntentAlignment(symmetryData.selected, playerIntent)
    }

    -- Aplica quebra de simetria
    local brokenSymmetry = self:applySymmetryBreak(
        symmetryData.selected,
        symmetryGroup.type,
        guidanceVector
    )

    symmetryData.result = brokenSymmetry

    return symmetryData
end

function SymmetryBreaker:evaluateState(state, guidanceVector, context)
    -- Avalia um estado candidato para quebra de simetria
    local evaluation = {
        state = state,
        coherenceScore = self:calculateCoherence(state, context.group),
        ethicalAlignment = self:calculateEthicalAlignment(state, guidanceVector),
        stability = self:calculateStability(state),
        novelty = self:calculateNovelty(state, context.group.previousStates),
        explorationPotential = self:calculateExplorationPotential(state)
    }

    -- Pontuação composta
    evaluation.compositeScore =
        evaluation.coherenceScore * 0.4 +
        evaluation.ethicalAlignment * 0.3 +
        evaluation.stability * 0.2 +
        evaluation.explorationPotential * 0.1

    return evaluation
end

function SymmetryBreaker:calculateEthicalAlignment(state, guidanceVector)
    -- Calcula alinhamento ético baseado no vetor de orientação
    local alignment = 0
    local dimensions = math.min(#state, #guidanceVector)

    for i = 1, dimensions do
        local stateVal = state[i] or 0
        local guideVal = guidanceVector[i] or 0
        alignment = alignment + (1 - math.abs(stateVal - guideVal))
    end

    return alignment / dimensions
end

function SymmetryBreaker:blendVectors(vector1, vector2, weight2)
    -- Mistura dois vetores com peso
    weight2 = weight2 or 0.3  -- Peso padrão para intenção do jogador
    local result = {}

    for i = 1, math.max(#vector1, #vector2) do
        local v1 = vector1[i] or 0
        local v2 = vector2[i] or 0
        result[i] = v1 * (1 - weight2) + v2 * weight2
    end

    return result
end

-- ============ QUANTUM ORACLE IMPLEMENTAÇÃO ============

local QuantumOracle = {
    _ontologicalStack = {
        {name = "Physical", depth = 1, coherence = 0.95, entropy = 0.1},
        {name = "Computational", depth = 2, coherence = 0.85, entropy = 0.3},
        {name = "Linguistic", depth = 3, coherence = 0.75, entropy = 0.5},
        {name = "Mathematical", depth = 4, coherence = 0.90, entropy = 0.2},
        {name = "Quantum", depth = 5, coherence = 0.70, entropy = 0.7},
        {name = "Consciousness", depth = 6, coherence = 0.65, entropy = 0.8},
        {name = "Cosmic", depth = 7, coherence = 0.80, entropy = 0.4}
    },
    _qrng = nil  -- Simulador de gerador quântico de números aleatórios
}

function QuantumOracle:initializeQRNG()
    -- Inicializa o gerador quântico de números aleatórios
    self._qrng = {
        lastValue = os.time(),
        entropyPool = {},
        quantumState = 0
    }

    -- Preenche pool de entropia
    for i = 1, 1000 do
        table.insert(self._qrng.entropyPool, math.random())
    end
end

function QuantumOracle:calculateLayerResonance(layer, seed)
    return 0.75
end

function QuantumOracle:calculateAggregateProbability(probs)
    local total = 0
    local count = 0
    for _, p in pairs(probs) do
        total = total + p
        count = count + 1
    end
    return total / count
end

function QuantumOracle:generateEmergenceRecommendations(analysis)
    return {"Accelerate quantum research"}
end

function QuantumOracle:generateConceptDescription(seed, transform, layer)
    return "A complex emergent concept"
end

function QuantumOracle:queryEmergenceProbability(conceptSeed, playerContext)
    -- Consulta a probabilidade de novas ideias emergirem
    local queryId = game:GetService("HttpService"):GenerateGUID(false)
    local queryData = {
        id = queryId,
        seed = conceptSeed,
        player = playerContext,
        timestamp = os.time(),
        stackAnalysis = {},
        probabilities = {}
    }

    -- Gera número aleatório quântico
    local quantumRandom = self:generateQuantumRandom()

    -- Analisa cada camada da pilha ontológica
    for i, layer in ipairs(self._ontologicalStack) do
        local layerAnalysis = {
            layer = layer.name,
            depth = layer.depth,
            baseCoherence = layer.coherence,
            currentEntropy = layer.entropy
        }

        -- Calcula probabilidade de emergência nesta camada
        local emergenceProb = self:calculateLayerEmergenceProbability(
            layer,
            quantumRandom,
            conceptSeed,
            playerContext
        )

        layerAnalysis.emergenceProbability = emergenceProb
        layerAnalysis.favorableConditions = emergenceProb > 0.5

        -- Mapeia para conceitos emergentes potenciais
        if emergenceProb > 0.3 then
            layerAnalysis.potentialConcepts = self:generatePotentialConcepts(
                conceptSeed,
                layer,
                emergenceProb,
                playerContext
            )
        end

        table.insert(queryData.stackAnalysis, layerAnalysis)
        queryData.probabilities[layer.name] = emergenceProb
    end

    -- Probabilidade agregada
    queryData.aggregateProbability = self:calculateAggregateProbability(queryData.probabilities)

    -- Recomendações baseadas nas camadas mais promissoras
    queryData.recommendations = self:generateEmergenceRecommendations(queryData.stackAnalysis)

    -- Mapeia o número aleatório quântico para conceitos específicos
    queryData.quantumMapping = self:mapQuantumRandomToConcepts(quantumRandom, conceptSeed)

    return queryData
end

function QuantumOracle:generateQuantumRandom()
    -- Gera um número aleatório "quântico"
    if not self._qrng then
        self:initializeQRNG()
    end

    -- Combina múltiplas fontes de aleatoriedade
    local timeEntropy = os.clock() % 1
    local poolEntropy = table.remove(self._qrng.entropyPool, 1) or math.random()
    local systemEntropy = math.random()

    -- Adiciona de volta ao pool
    table.insert(self._qrng.entropyPool, systemEntropy)

    -- Aplica transformação não-linear
    local quantumValue = (timeEntropy * 0.3 + poolEntropy * 0.4 + systemEntropy * 0.3)
    quantumValue = math.sin(quantumValue * math.pi * 2) * 0.5 + 0.5  -- Mapeia para [0,1]

    self._qrng.lastValue = quantumValue
    self._qrng.quantumState = (self._qrng.quantumState + quantumValue) % 1

    return quantumValue
end

function QuantumOracle:calculateLayerEmergenceProbability(layer, quantumRandom, conceptSeed, playerContext)
    -- Calcula probabilidade de emergência em uma camada específica
    local baseProb = layer.coherence * (1 - layer.entropy)

    -- Modificadores baseados no contexto
    local modifiers = {
        quantum = quantumRandom * 0.3,
        seedComplexity = #tostring(conceptSeed) / 100,
        playerCoherence = playerContext.coherence or 0.5,
        layerResonance = self:calculateLayerResonance(layer, conceptSeed)
    }

    local totalModifier = 0
    for _, mod in pairs(modifiers) do
        totalModifier = totalModifier + mod
    end
    totalModifier = totalModifier / 4

    -- Probabilidade final
    local probability = baseProb * (0.7 + totalModifier * 0.3)

    return math.min(0.95, math.max(0.05, probability))
end

function QuantumOracle:generatePotentialConcepts(seed, layer, probability, playerContext)
    -- Gera conceitos emergentes potenciais baseados na semente e camada
    local concepts = {}

    -- Transformações baseadas na camada
    local transformations = {
        Physical = {"materialize", "embody", "solidify"},
        Computational = {"compute", "simulate", "optimize"},
        Linguistic = {"articulate", "narrate", "semanticize"},
        Mathematical = {"quantify", "structure", "formalize"},
        Quantum = {"superpose", "entangle", "cohere"},
        Consciousness = {"perceive", "intuit", "realize"},
        Cosmic = {"expand", "transcend", "unify"}
    }

    local layerTransforms = transformations[layer.name] or {"transform", "evolve", "emerge"}

    for i, transform in ipairs(layerTransforms) do
        local concept = {
            name = transform .. "_" .. seed:sub(1, 5) .. "_" .. layer.name:sub(1, 3),
            transformation = transform,
            sourceLayer = layer.name,
            probability = probability * (0.8 + i * 0.1),
            requiredCoherence = layer.coherence * 0.8,
            description = self:generateConceptDescription(seed, transform, layer)
        }

        table.insert(concepts, concept)
    end

    return concepts
end

function QuantumOracle:mapQuantumRandomToConcepts(quantumRandom, seed)
    -- Mapeia o número aleatório quântico para conceitos específicos
    local mappings = {
        [0.0] = "void_concept",
        [0.1] = "primordial_idea",
        [0.2] = "structured_thought",
        [0.3] = "complex_system",
        [0.4] = "living_concept",
        [0.5] = "conscious_idea",
        [0.6] = "transcendent_concept",
        [0.7] = "cosmic_principle",
        [0.8] = "unified_theory",
        [0.9] = "divine_insight"
    }

    -- Encontra o bucket mais próximo
    local bucket = math.floor(quantumRandom * 10) / 10
    local baseConcept = mappings[bucket] or "emergent_idea"

    return {
        quantumValue = quantumRandom,
        mappedConcept = baseConcept,
        specificConcept = baseConcept .. "_" .. seed:sub(1, 3),
        certainty = 0.7 + (quantumRandom % 0.3)
    }
end

-- ============ COSMOPSYCHIA SERVICE COM HEALTH CHECK ============

local CosmopsychiaService = {
    _layers = {
        physical_layer = {
            name = "Physical",
            coherence = 0.95,
            stability = 0.92,
            parameters = {
                gravity = 9.8,
                lightspeed = 299792458,
                planck_length = 1.616e-35
            },
            metrics = {
                update_rate = 60,  -- Hz
                collision_accuracy = 0.99,
                render_stability = 0.98
            }
        },

        computational_layer = {
            name = "Computational",
            coherence = 0.85,
            stability = 0.88,
            parameters = {
                quantum_bit_depth = 64,
                processing_speed = 1e12,  -- ops/sec
                memory_coherence = 0.95
            },
            metrics = {
                computation_throughput = 0.9,
                algorithm_efficiency = 0.85,
                error_correction_rate = 0.99
            }
        },

        language_syntax_layer = {
            name = "Linguistic Syntax",
            coherence = 0.75,
            stability = 0.82,
            parameters = {
                grammatical_complexity = 0.7,
                semantic_density = 0.8,
                expressive_power = 0.9
            },
            metrics = {
                parsing_accuracy = 0.95,
                translation_fidelity = 0.88,
                conceptual_clarity = 0.8
            }
        }
    },

    _healthThresholds = {
        critical = 0.4,
        warning = 0.7,
        healthy = 0.85
    }
}

-- Mocks for missing methods in CosmopsychiaService
function CosmopsychiaService:calculate_variance(list, field)
    return 0.05
end

function CosmopsychiaService:calculate_layer_correlation(reports)
    return 0.92
end

function CosmopsychiaService:analyze_health_trend()
    return "Stable"
end

function CosmopsychiaService:identify_secondary_issues(layerData)
    return {}
end

function CosmopsychiaService:estimate_recovery_rate(layerData)
    return 0.15
end

function CosmopsychiaService:check_substrate_health()
    -- Analisa a saúde de todas as camadas do substrato
    local healthReport = {
        timestamp = os.time(),
        overallHealth = 0,
        layerReports = {},
        criticalIssues = {},
        recommendations = [],
        metrics = {}
    }

    local totalCoherence = 0
    local totalStability = 0
    local layerCount = 0

    -- Analisa cada camada individualmente
    for layerName, layerData in pairs(self._layers) do
        local layerReport = self:analyze_layer_health(layerName, layerData)
        table.insert(healthReport.layerReports, layerReport)

        totalCoherence = totalCoherence + layerReport.coherence
        totalStability = totalStability + layerReport.stability
        layerCount = layerCount + 1

        -- Identifica problemas críticos
        if layerReport.healthStatus == "critical" then
            table.insert(healthReport.criticalIssues, {
                layer = layerName,
                issue = layerReport.primaryIssue,
                severity = layerReport.healthScore
            })
        end
    end

    -- Calcula métricas gerais
    healthReport.overallHealth = (totalCoherence + totalStability) / (2 * layerCount)
    healthReport.averageCoherence = totalCoherence / layerCount
    healthReport.averageStability = totalStability / layerCount

    -- Gera recomendações baseadas nos problemas identificados
    healthReport.recommendations = self:generate_health_recommendations(healthReport)

    -- Classifica o estado geral
    healthReport.systemStatus = self:classify_system_status(healthReport.overallHealth)

    -- Coleta métricas detalhadas
    healthReport.metrics = {
        coherence_variance = self:calculate_variance(healthReport.layerReports, "coherence"),
        stability_variance = self:calculate_variance(healthReport.layerReports, "stability"),
        layer_correlation = self:calculate_layer_correlation(healthReport.layerReports),
        trend = self:analyze_health_trend()
    }

    return healthReport
end

function CosmopsychiaService:classify_system_status(health)
    if health > 0.8 then return "healthy"
    elseif health > 0.5 then return "warning"
    else return "critical" end
end

function CosmopsychiaService:analyze_layer_health(layerName, layerData)
    -- Analisa a saúde de uma camada específica
    local report = {
        layer = layerName,
        name = layerData.name,
        coherence = layerData.coherence,
        stability = layerData.stability,
        parameters = layerData.parameters,
        metrics = layerData.metrics
    }

    -- Calcula pontuação de saúde
    local healthScore = (layerData.coherence * 0.6 + layerData.stability * 0.4)
    report.healthScore = healthScore

    -- Identifica o problema primário
    report.primaryIssue = self:identify_primary_issue(layerData)
    report.secondaryIssues = self:identify_secondary_issues(layerData)

    -- Classifica o estado de saúde
    if healthScore < self._healthThresholds.critical then
        report.healthStatus = "critical"
        report.recoveryPriority = "immediate"
    elseif healthScore < self._healthThresholds.warning then
        report.healthStatus = "warning"
        report.recoveryPriority = "high"
    elseif healthScore < self._healthThresholds.healthy then
        report.healthStatus = "moderate"
        report.recoveryPriority = "medium"
    else
        report.healthStatus = "healthy"
        report.recoveryPriority = "low"
    end

    -- Calcula resiliência da camada
    report.resilience = self:calculate_layer_resilience(layerData)

    -- Sugestões de recuperação específicas
    report.recoverySuggestions = self:generate_layer_recovery_suggestions(
        layerName,
        layerData,
        report.primaryIssue
    )

    return report
end

function CosmopsychiaService:identify_primary_issue(layerData)
    -- Identifica o problema mais crítico em uma camada
    local issues = {}

    if layerData.coherence < 0.6 then
        table.insert(issues, "low_coherence")
    elseif layerData.coherence < 0.8 then
        table.insert(issues, "moderate_coherence")
    end

    if layerData.stability < 0.7 then
        table.insert(issues, "instability")
    end

    -- Verifica parâmetros específicos por camada
    if layerData.name == "Physical" then
        if layerData.parameters.gravity < 9.7 or layerData.parameters.gravity > 9.9 then
            table.insert(issues, "gravity_anomaly")
        end
    elseif layerData.name == "Computational" then
        if layerData.metrics.error_correction_rate < 0.98 then
            table.insert(issues, "high_error_rate")
        end
    elseif layerData.name == "Linguistic Syntax" then
        if layerData.metrics.conceptual_clarity < 0.75 then
            table.insert(issues, "conceptual_ambiguity")
        end
    end

    return issues[1] or "none_detected"
end

function CosmopsychiaService:generate_layer_recovery_suggestions(layerName, layerData, primaryIssue)
    -- Gera sugestões de recuperação específicas para cada tipo de problema
    local suggestions = {}

    if primaryIssue == "low_coherence" then
        if layerName == "physical_layer" then
            table.insert(suggestions, "Run geometric consistency checks")
            table.insert(suggestions, "Increase entanglement stabilization")
            table.insert(suggestions, "Reduce observer interference")
        elseif layerName == "computational_layer" then
            table.insert(suggestions, "Optimize quantum algorithms")
            table.insert(suggestions, "Increase error correction cycles")
            table.insert(suggestions, "Defragment quantum memory")
        elseif layerName == "language_syntax_layer" then
            table.insert(suggestions, "Simplify grammatical structures")
            table.insert(suggestions, "Increase semantic clarity")
            table.insert(suggestions, "Reduce ambiguous constructs")
        end
    elseif primaryIssue == "instability" then
        table.insert(suggestions, "Implement stability buffers")
        table.insert(suggestions, "Increase update frequency")
        table.insert(suggestions, "Add redundancy checks")
    elseif primaryIssue == "gravity_anomaly" then
        table.insert(suggestions, "Recalibrate gravitational constants")
        table.insert(suggestions, "Check for mass distribution anomalies")
        table.insert(suggestions, "Verify curvature tensor calculations")
    end

    -- Adiciona sugestões gerais baseadas no score
    if layerData.coherence < 0.7 then
        table.insert(suggestions, "Consider temporary layer quarantine")
    end

    if layerData.stability < 0.75 then
        table.insert(suggestions, "Implement gradual stabilization protocol")
    end

    return suggestions
end

function CosmopsychiaService:generate_health_recommendations(healthReport)
    -- Gera recomendações gerais baseadas no relatório de saúde
    local recommendations = {}

    if healthReport.overallHealth < self._healthThresholds.critical then
        table.insert(recommendations, "IMMEDIATE ACTION REQUIRED: System coherence critical")
        table.insert(recommendations, "Initiate emergency stabilization protocols")
        table.insert(recommendations, "Isolate affected layers to prevent cascade failure")
    elseif healthReport.overallHealth < self._healthThresholds.warning then
        table.insert(recommendations, "System showing signs of instability")
        table.insert(recommendations, "Increase monitoring frequency")
        table.insert(recommendations, "Prepare contingency plans for critical layers")
    end

    -- Recomendações específicas baseadas nas camadas problemáticas
    for _, layerReport in ipairs(healthReport.layerReports) do
        if layerReport.healthStatus == "critical" then
            table.insert(recommendations,
                "Priority recovery for " .. layerReport.name .. " layer: " ..
                table.concat(layerReport.recoverySuggestions, "; "))
        elseif layerReport.healthStatus == "warning" then
            table.insert(recommendations,
                "Monitor " .. layerReport.name .. " layer closely: " ..
                layerReport.primaryIssue)
        end
    end

    -- Recomendações preventivas
    table.insert(recommendations, "Schedule regular coherence calibration sessions")
    table.insert(recommendations, "Maintain entanglement network optimization")
    table.insert(recommendations, "Keep quantum error correction algorithms updated")

    return recommendations
end

function CosmopsychiaService:calculate_layer_resilience(layerData)
    -- Calcula a resiliência de uma camada baseada em múltiplos fatores
    local resilience = 0

    -- Componente de coerência
    local coherenceComponent = layerData.coherence * 0.4

    -- Componente de estabilidade
    local stabilityComponent = layerData.stability * 0.3

    -- Componente de redundância (estimada)
    local redundancy = 0.7  -- Valor padrão, poderia ser calculado
    local redundancyComponent = redundancy * 0.2

    -- Componente de recuperação
    local recoveryRate = self:estimate_recovery_rate(layerData)
    local recoveryComponent = recoveryRate * 0.1

    resilience = coherenceComponent + stabilityComponent +
                 redundancyComponent + recoveryComponent

    return resilience
end

-- ============ INICIALIZAÇÃO DO SISTEMA AVANÇADO ============

local AdvancedqRoblox = {
    OntologicalKernel = OntologicalKernel,
    UniversalCompiler = UniversalCompiler,
    CosmopsychiaVisualizer = CosmopsychiaVisualizer,
    SymmetryBreaker = SymmetryBreaker,
    QuantumOracle = QuantumOracle,
    CosmopsychiaService = CosmopsychiaService,
    _playerConnections = {}
}

function AdvancedqRoblox:handleCriticalHealthIssues(report)
    print("Handled critical issues")
end

function AdvancedqRoblox:setupAdvancedServices()
    print("Setup advanced services")
end

function AdvancedqRoblox:initialize()
    print("🚀 Inicializando qRoblox Advanced Edition...")

    -- Inicializa todos os módulos
    self.QuantumOracle:initializeQRNG()

    -- Configura serviços avançados
    self:setupAdvancedServices()

    -- Inicia monitoramento de saúde
    self:startHealthMonitoring()

    -- Configura visualizadores para jogadores
    self:setupPlayerVisualizers()

    print("✅ qRoblox Advanced Edition inicializado!")
    print("   Módulos carregados:")
    print("   • OntologicalKernel com tratamento de erros avançado")
    print("   • UniversalCompiler com sugestões de correção")
    print("   • CosmopsychiaVisualizer com D3.js")
    print("   • SymmetryBreaker com vetores de orientação")
    print("   • QuantumOracle com pilha ontológica de 7 camadas")
    print("   • CosmopsychiaService com health check completo")

    return true
end

function AdvancedqRoblox:startHealthMonitoring()
    -- Inicia monitoramento periódico da saúde do sistema
    local heartbeat = game:GetService("RunService").Heartbeat:Connect(function()
        -- Verificação de saúde a cada 30 segundos
        if os.time() % 30 == 0 then
            local healthReport = self.CosmopsychiaService:check_substrate_health()

            if healthReport.systemStatus == "critical" then
                warn("🚨 CRITICAL SYSTEM HEALTH: " .. healthReport.overallHealth)
                warn("   Issues: " .. #healthReport.criticalIssues)

                -- Ações automáticas para problemas críticos
                self:handleCriticalHealthIssues(healthReport)
            end
        end
    end)

    -- Armazena a conexão para limpeza posterior
    self._healthMonitor = heartbeat
end

function AdvancedqRoblox:setupPlayerVisualizers()
    -- Configura visualizadores para novos jogadores
    Players.PlayerAdded:Connect(function(player)
        player.CharacterAdded:Wait()

        -- Cria visualizador para o jogador
        local visualizer = self.CosmopsychiaVisualizer:createVisualizer(player, "forceDirected")

        -- Atualiza periodicamente com dados do sistema
        local updateConnection = game:GetService("RunService").Heartbeat:Connect(function()
            local data = self.CosmopsychiaVisualizer:generateCosmopsychiaData(player)
            self.CosmopsychiaVisualizer:updateVisualization(player, data)
        end)

        -- Armazena para limpeza
        self._playerConnections[player.UserId] = updateConnection
    end)
end

return AdvancedqRoblox
