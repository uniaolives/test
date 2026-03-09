// orb_service.go
package main

import "fmt"

type Orb struct {
    Stability float64
    Frequency float64
}

func NewOrb(lambda float64, freq float64) (*Orb, bool) {
    if lambda > 0.618 {
        return &Orb{Stability: lambda, Frequency: freq}, true
    }
    return nil, false
}

func (o *Orb) Transmit(handover string) error {
    if o.Stability > 0.5 {
        fmt.Printf("Transmitting via Orb: %s\n", handover)
        return nil
    }
    return fmt.Errorf("wormhole collapsed")
}
