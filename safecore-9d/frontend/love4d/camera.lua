-- safecore-9d/frontend/love4d/camera.lua
-- Hyperbolic Tesseract Projection logic
-- Projects 33-dimensional coordinates onto a 2D screen

local Camera = {}
Camera.__index = Camera

function Camera.new(dims)
    local self = setmetatable({}, Camera)
    self.dims = dims or 33
    self.angle = 0
    -- Rotation matrix for 33D space to 2D projection
    self.matrix = {}
    for i = 1, self.dims do
        self.matrix[i] = {
            (math.random() - 0.5) * 2,
            (math.random() - 0.5) * 2
        }
    end
    return self
end

function Camera:project(coords, scale)
    local x, y = 0, 0
    scale = scale or 200

    local w = 800 -- Default width if love is not present
    local h = 600 -- Default height
    if love and love.graphics then
        w = love.graphics.getWidth()
        h = love.graphics.getHeight()
    end

    -- Project N-dims to 2D via the rotation matrix
    for i = 1, math.min(#coords, self.dims) do
        x = x + coords[i] * self.matrix[i][1]
        y = y + coords[i] * self.matrix[i][2]
    end

    -- Apply Hyperbolic Scaling:
    -- Points further out get exponentially squashed (Poincaré Disk feel)
    local r = math.sqrt(x*x + y*y)
    local factor = 1 / (1 + r * 0.1)

    return x * scale * factor + w/2,
           y * scale * factor + h/2
end

return Camera
