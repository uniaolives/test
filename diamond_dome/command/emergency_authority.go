package command

import (
	"errors"
	"log"
)

type Threat struct {
	ID        string
	Origin    string
	Trajectory [3]float64
}

type EmergencyAuthority struct {
	CurrentPhi float64
}

func (ea *EmergencyAuthority) AuthorizeEngagement(target Threat) error {
	// P1 - Soberania
	if target.Origin == "UNKNOWN" {
		return errors.New("engagement vetoed: origin unknown")
	}

	if ea.CurrentPhi > 0.8 {
		return errors.New("engagement vetoed: system instability (phi too high)")
	}

	log.Printf("ENGAGEMENT_AUTHORIZED: Target %s", target.ID)
	return nil
}
