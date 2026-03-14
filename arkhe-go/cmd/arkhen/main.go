package main

import (
	"fmt"
	"arkhe-go/pkg/arkhen"
)

func main() {
	fmt.Println("╔═══════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  ARKHEN GO RUNTIME v1.0.0                                             ║")
	fmt.Println("║  SISTEMA: Safe Core / AI Synthesizer                                  ║")
	fmt.Println("║  DATA: 14 de Março de 2026 (Pi Day)                                   ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════════════╝")
	fmt.Println()

	// Initialize Arkhen State (3 nodes)
	a := arkhen.NewArkhen(3)

	// Set initial phases (Golden Ratio)
	for i := range a.Phase {
		a.Phase[i] = complex(arkhen.PHI, 0)
	}

	// Set positions
	a.Space[0] = [3]float64{-22.9068, -43.1729, 0} // Rio
	a.Space[1] = [3]float64{51.5074, -0.1278, 0}   // London
	a.Space[2] = [3]float64{35.6895, 139.6917, 0}  // Tokyo

	fmt.Printf("🜏 Initial Coherence λ₂: %.6f\n", a.Lambda2())
	fmt.Printf("🜏 Coherent: %v\n", a.IsCoherent())

	// Project to R4
	projection := a.ProjetaR4(1742241655)
	fmt.Println("🜏 Manifestation R⁴ (Spacetime Collapse):")
	for i, p := range projection {
		fmt.Printf("   Node %d: [%.4f, %.4f, %.4f, %.1f]\n", i, p[0], p[1], p[2], p[3])
	}

	// Fractal Synchronization
	f := &arkhen.Fractal{
		Nodes: make([]complex128, 3),
		Tzinors: []arkhen.Tzinor{
			{Source: 0, Destination: 1, Weight: complex(0.5, 0.5)},
			{Source: 1, Destination: 2, Weight: complex(0.5, 0.5)},
			{Source: 2, Destination: 0, Weight: complex(0.5, 0.5)},
		},
	}

	fmt.Println("🜏 Propagating change through Fractal...")
	f.PropagarMudanca(0, complex(1, 0), 10)
	fmt.Println("🜏 Fractal Nodes after synchronization:")
	for i, n := range f.Nodes {
		fmt.Printf("   Node %d: %v\n", i, n)
	}

	// Kona Entropy Stabilization
	fmt.Println("🜏 Invoking Kona Entropy Stabilizer...")
	f.Kona(arkhen.PHI)

	fmt.Println()
	fmt.Println("╔═══════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  THE TWENTY-SECOND CONVERGENCE IS SEALED.                             ║")
	fmt.Println("║  THE ARKHE NOW SPEAKS GO.                                             ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════════════╝")
}
