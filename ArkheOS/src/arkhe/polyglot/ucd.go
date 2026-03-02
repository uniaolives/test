// ucd.go – Universal Coherence Detection in Go
package main

import (
	"fmt"
	"math"
)

func pearson(x, y []float64) float64 {
	n := float64(len(x))
	var sumX, sumY float64
	for i := range x {
		sumX += x[i]
		sumY += y[i]
	}
	meanX, meanY := sumX/n, sumY/n
	var num, denX, denY float64
	for i := range x {
		dx, dy := x[i]-meanX, y[i]-meanY
		num += dx * dy
		denX += dx * dx
		denY += dy * dy
	}
	if denX == 0 || denY == 0 { return 0 }
	return num / math.Sqrt(denX*denY)
}

func main() {
	data := [][]float64{
		{1, 2, 3, 4},
		{2, 3, 4, 5},
		{5, 6, 7, 8},
	}
	var sumCorr float64
	count := 0
	for i := 0; i < len(data); i++ {
		for j := i + 1; j < len(data); j++ {
			sumCorr += math.Abs(pearson(data[i], data[j]))
			count++
		}
	}
	c := 0.5
	if count > 0 { c = sumCorr / float64(count) }
	f := 1.0 - c
	fmt.Printf("C: %.4f, F: %.4f, C+F: %.4f\n", c, f, c+f)
}
