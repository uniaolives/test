package main

import (
	"fmt"
	"math"
	"time"
)

// Human represents a human user with cognitive limits.
type Human struct {
	ProcessingCapacity float64  // bits/min
	AttentionSpan      float64  // minutes
	CurrentLoad        float64  // 0.0 to 1.0
	Goals              []string
}

// Tool represents an AI tool.
type Tool struct {
	OutputVolume      float64 // tokens/min
	OutputEntropy     float64 // bits/token
	HasDiscernment    bool    // Always false
	HasIntentionality bool    // Always false
	HasPerception     bool    // Always false
}

// LogEvent records interaction details.
type LogEvent struct {
	Time     time.Time
	Event    string
	Load     float64
	Intent   string
	Approved bool
	Output   string
}

// InteractionGuard monitors and protects the human-tool relationship.
type InteractionGuard struct {
	Human     *Human
	Tool      *Tool
	Log       []LogEvent
	Threshold float64
}

// NewInteractionGuard creates a new guard.
func NewInteractionGuard(h *Human, t *Tool) *InteractionGuard {
	return &InteractionGuard{
		Human:     h,
		Tool:      t,
		Log:       make([]LogEvent, 0),
		Threshold: 0.7,
	}
}

// ProposeInteraction checks if the interaction is safe.
func (g *InteractionGuard) ProposeInteraction(intent string) string {
	load := (g.Tool.OutputVolume * g.Tool.OutputEntropy) / g.Human.ProcessingCapacity

	if load > g.Threshold {
		g.Log = append(g.Log, LogEvent{
			Time:  time.Now(),
			Event: "BLOCKED",
			Load:  load,
		})
		return ""
	}

	if g.Human.CurrentLoad > 0.8 {
		g.Log = append(g.Log, LogEvent{
			Time:  time.Now(),
			Event: "BLOCKED",
			Load:  g.Human.CurrentLoad,
		})
		return ""
	}

	// Generate content (simulated)
	output := fmt.Sprintf("Generated content for: %s", intent)

	impact := load * 0.3
	g.Human.CurrentLoad = math.Min(1.0, g.Human.CurrentLoad+impact)

	g.Log = append(g.Log, LogEvent{
		Time:   time.Now(),
		Event:  "GENERATED",
		Load:   load,
		Intent: intent,
	})

	return output
}

// Review processes human approval or rejection.
func (g *InteractionGuard) Review(output string, approved bool) {
	g.Log = append(g.Log, LogEvent{
		Time:     time.Now(),
		Event:    "REVIEWED",
		Approved: approved,
		Output:   output,
	})

	if approved {
		g.Human.CurrentLoad = math.Max(0.0, g.Human.CurrentLoad-0.1)
	}
}

// CognitiveLoadIndex calculates ISC.
func (g *InteractionGuard) CognitiveLoadIndex(windowMinutes int) float64 {
	cutoff := time.Now().Add(time.Duration(-windowMinutes) * time.Minute)
	var recentCount, overloadCount int

	for _, e := range g.Log {
		if e.Time.After(cutoff) {
			recentCount++
			if e.Load > g.Threshold {
				overloadCount++
			}
		}
	}

	if recentCount == 0 {
		return 0.0
	}
	return float64(overloadCount) / float64(recentCount)
}

// AuthorshipLossRate calculates TPA.
func (g *InteractionGuard) AuthorshipLossRate(windowMinutes int) float64 {
	cutoff := time.Now().Add(time.Duration(-windowMinutes) * time.Minute)
	var reviewCount, totalCount int

	for _, e := range g.Log {
		if e.Time.After(cutoff) {
			if e.Event == "REVIEWED" {
				reviewCount++
			}
			if e.Event == "GENERATED" || e.Event == "REVIEWED" {
				totalCount++
			}
		}
	}

	if totalCount == 0 {
		return 0.0
	}
	return float64(reviewCount) / float64(totalCount)
}

func main() {
	human := &Human{ProcessingCapacity: 500, AttentionSpan: 30}
	tool := &Tool{OutputVolume: 100, OutputEntropy: 2.5}
	guard := NewInteractionGuard(human, tool)

	fmt.Println("Testing Arkhe Human-Tool Interface in Go...")
	output := guard.ProposeInteraction("Write a summary of the Arkhe Protocol")
	if output != "" {
		fmt.Printf("✅ Generated: %s\n", output)
		guard.Review(output, true)
	}

	fmt.Printf("📊 ISC: %.3f\n", guard.CognitiveLoadIndex(60))
	fmt.Printf("📊 TPA: %.3f\n", guard.AuthorshipLossRate(60))
}
