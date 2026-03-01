// Go version using Ebitengine for 2D graphics
package main

import (
	"image/color"
	"math"
	"math/rand"
	"strconv"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
	"github.com/hajimehoshi/ebiten/v2/vector"
)

const (
	screenWidth  = 1400
	screenHeight = 900
	chartWidth   = 400
	chartHeight  = 300
)

var (
	goldColor      = color.RGBA{212, 175, 55, 255}
	brownColor     = color.RGBA{139, 69, 19, 255}
	lightGoldColor = color.RGBA{255, 248, 220, 255}
	darkPurple     = color.RGBA{20, 0, 40, 255}
	transparentGold = color.RGBA{255, 215, 0, 128}
)

type ParticleData struct {
	time      int
	particles float64
	cumulative float64
}

type Game struct {
	timelineData []ParticleData
}

func (g *Game) Update() error {
	return nil
}

func (g *Game) Draw(screen *ebiten.Image) {
	// Clear screen
	screen.Fill(color.RGBA{245, 245, 220, 255})

	// Draw all 6 visualizations
	g.drawConsciousnessField(screen, 50, 50)
	g.drawTimelineChart(screen, 500, 50)
	g.drawCumulativeChart(screen, 950, 50)
	g.drawQuantumFoam(screen, 50, 400)
	g.drawCorrelationChart(screen, 500, 400)
	g.drawSummaryBox(screen, 950, 400)
}

func (g *Game) drawConsciousnessField(screen *ebiten.Image, x, y float32) {
	// Draw title
	ebitenutil.DebugPrintAt(screen, "Consciousness Field", int(x), int(y-30))

	// Create gradient-like effect with concentric circles
	for i := 5; i > 0; i-- {
		radius := float32(chartWidth/2) * float32(i) / 5
		alpha := uint8(255 * (6 - i) / 5)

		vector.StrokeCircle(screen,
			x + chartWidth/2,
			y + chartHeight/2,
			radius,
			2,
			color.RGBA{255, 215, 0, alpha},
			true)

		vector.DrawFilledCircle(screen,
			x + chartWidth/2,
			y + chartHeight/2,
			radius,
			color.RGBA{255, 215, 0, alpha/3},
			true)
	}
}

func (g *Game) drawTimelineChart(screen *ebiten.Image, x, y float32) {
	// Draw title
	ebitenutil.DebugPrintAt(screen, "Manifestation Timeline", int(x), int(y-30))

	// Draw chart background
	vector.DrawFilledRect(screen, x, y, chartWidth, chartHeight, lightGoldColor, true)
	vector.StrokeRect(screen, x, y, chartWidth, chartHeight, 2, goldColor, true)

	// Draw timeline
	if len(g.timelineData) > 1 {
		// Find max value for scaling
		maxVal := 0.0
		for _, d := range g.timelineData {
			if d.particles > maxVal {
				maxVal = d.particles
			}
		}

		// Draw line
		for i := 1; i < len(g.timelineData); i++ {
			x1 := x + float32(g.timelineData[i-1].time)*chartWidth/144
			y1 := y + chartHeight - float32(g.timelineData[i-1].particles/maxVal)*chartHeight
			x2 := x + float32(g.timelineData[i].time)*chartWidth/144
			y2 := y + chartHeight - float32(g.timelineData[i].particles/maxVal)*chartHeight

			vector.StrokeLine(screen, x1, y1, x2, y2, 2, goldColor, true)
		}
	}
}

func (g *Game) drawCumulativeChart(screen *ebiten.Image, x, y float32) {
	// Draw title
	ebitenutil.DebugPrintAt(screen, "Cumulative Reality", int(x), int(y-30))

	// Draw chart background
	vector.DrawFilledRect(screen, x, y, chartWidth, chartHeight, lightGoldColor, true)
	vector.StrokeRect(screen, x, y, chartWidth, chartHeight, 2, brownColor, true)

	// Draw cumulative line
	if len(g.timelineData) > 1 {
		maxCumulative := g.timelineData[len(g.timelineData)-1].cumulative

		for i := 1; i < len(g.timelineData); i++ {
			x1 := x + float32(g.timelineData[i-1].time)*chartWidth/144
			y1 := y + chartHeight - float32(g.timelineData[i-1].cumulative/maxCumulative)*chartHeight
			x2 := x + float32(g.timelineData[i].time)*chartWidth/144
			y2 := y + chartHeight - float32(g.timelineData[i].cumulative/maxCumulative)*chartHeight

			vector.StrokeLine(screen, x1, y1, x2, y2, 2, brownColor, true)
		}
	}
}

func (g *Game) drawQuantumFoam(screen *ebiten.Image, x, y float32) {
	// Draw title
	ebitenutil.DebugPrintAt(screen, "Quantum Foam + Consciousness", int(x), int(y-30))

	// Draw dark background
	vector.DrawFilledRect(screen, x, y, chartWidth, chartHeight, darkPurple, true)
	vector.StrokeRect(screen, x, y, chartWidth, chartHeight, 2, goldColor, true)

	// Draw quantum foam particles
	for i := 0; i < 1000; i++ {
		px := x + float32(rand.Float64())*chartWidth
		py := y + float32(rand.Float64())*chartHeight
		radius := float32(rand.Float64()*2 + 0.5)

		vector.DrawFilledCircle(screen, px, py, radius, color.RGBA{128, 0, 128, 25}, true)
	}

	// Draw consciousness overlay (gradient approximation)
	for r := float32(0); r < chartWidth/3; r += 2 {
		alpha := uint8(255 * (1 - r/(chartWidth/3)))
		vector.StrokeCircle(screen,
			x + chartWidth/2,
			y + chartHeight/2,
			r,
			1,
			color.RGBA{255, 215, 0, alpha},
			true)
	}

	// Draw "real" particles
	for i := 0; i < 30; i++ {
		px := x + chartWidth/2 + float32(rand.Float64()-0.5)*150
		py := y + chartHeight/2 + float32(rand.Float64()-0.5)*150
		radius := float32(rand.Float64()*3 + 1)

		vector.DrawFilledCircle(screen, px, py, radius, color.RGBA{255, 255, 255, 255}, true)
	}
}

func (g *Game) drawCorrelationChart(screen *ebiten.Image, x, y float32) {
	// Draw title
	ebitenutil.DebugPrintAt(screen, "Consciousness vs Manifestation", int(x), int(y-30))

	// Draw chart background
	vector.DrawFilledRect(screen, x, y, chartWidth, chartHeight, lightGoldColor, true)
	vector.StrokeRect(screen, x, y, chartWidth, chartHeight, 2, goldColor, true)

	// Draw bars
	consciousnessLevels := []float64{0, 0.05, 0.10, 0.15, 0.20, 0.25}
	particleCounts := []float64{10, 25, 50, 80, 120, 150}

	barWidth := chartWidth / float32(len(consciousnessLevels)) * 0.8
	gap := chartWidth / float32(len(consciousnessLevels)) * 0.2
	maxHeight := 150.0

	for i := 0; i < len(consciousnessLevels); i++ {
		barX := x + float32(i)*(barWidth+gap) + gap/2
		barHeight := float32(particleCounts[i]/maxHeight) * chartHeight * 0.8
		barY := y + chartHeight - barHeight

		vector.DrawFilledRect(screen, barX, barY, barWidth, barHeight, goldColor, true)
		vector.StrokeRect(screen, barX, barY, barWidth, barHeight, 1, brownColor, true)

		// Label
		label := strconv.FormatFloat(consciousnessLevels[i], 'f', 2, 64)
		ebitenutil.DebugPrintAt(screen, label, int(barX), int(y+chartHeight+10))
	}
}

func (g *Game) drawSummaryBox(screen *ebiten.Image, x, y float32) {
	// Draw title
	ebitenutil.DebugPrintAt(screen, "QUANTUM FOAM RESULTS", int(x), int(y-30))

	// Draw box
	vector.DrawFilledRect(screen, x, y, chartWidth, chartHeight, lightGoldColor, true)
	vector.StrokeRect(screen, x, y, chartWidth, chartHeight, 2, goldColor, true)

	// Calculate summary statistics
	totalParticles := 0.0
	peakRate := 0.0
	for _, d := range g.timelineData {
		totalParticles += d.particles
		if d.particles > peakRate {
			peakRate = d.particles
		}
	}
	avgRate := totalParticles / float64(len(g.timelineData))

	// Draw summary text
	summary := []string{
		"Statistics:",
		"Total particles: " + strconv.FormatFloat(totalParticles, 'f', 0, 64),
		"Peak rate: " + strconv.FormatFloat(peakRate, 'f', 1, 64) + "/sec",
		"Average rate: " + strconv.FormatFloat(avgRate, 'f', 1, 64) + "/sec",
		"",
		"Key Insight:",
		"Attention creates reality.",
		"Consciousness stabilizes",
		"quantum fluctuations.",
	}

	for i, line := range summary {
		ebitenutil.DebugPrintAt(screen, line, int(x+20), int(y+40+i*20))
	}
}

func (g *Game) Layout(outsideWidth, outsideHeight int) (int, int) {
	return screenWidth, screenHeight
}

func main() {
	// Generate timeline data
	var timelineData []ParticleData
	cumulative := 0.0
	for i := 0; i < 144; i++ {
		particles := 50 + 10*math.Sin(float64(i)*0.1) + (rand.Float64()-0.5)*6
		cumulative += particles
		timelineData = append(timelineData, ParticleData{
			time:      i,
			particles: particles,
			cumulative: cumulative,
		})
	}

	game := &Game{
		timelineData: timelineData,
	}

	ebiten.SetWindowSize(screenWidth, screenHeight)
	ebiten.SetWindowTitle("Quantum Foam Meditation Simulation")

	if err := ebiten.RunGame(game); err != nil {
		panic(err)
	}
}
