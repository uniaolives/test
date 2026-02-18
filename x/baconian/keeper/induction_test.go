package keeper

import (
	"testing"

	"cosmossdk.io/math"
	"arkhend/x/baconian/types"
)

func TestFilter(t *testing.T) {
	observations := []types.Observation{
		{Context: "A", Result: true, Intensity: math.LegacyNewDec(1)},
		{Context: "B", Result: false, Intensity: math.LegacyNewDec(0)},
		{Context: "C", Result: true, Intensity: math.LegacyNewDecWithPrec(5, 1)},
	}

	presence := filter(observations, func(o types.Observation) bool { return o.Result })
	if len(presence) != 2 {
		t.Errorf("Expected 2 presence observations, got %d", len(presence))
	}

	absence := filter(observations, func(o types.Observation) bool { return !o.Result })
	if len(absence) != 1 {
		t.Errorf("Expected 1 absence observation, got %d", len(absence))
	}

	degrees := filter(observations, func(o types.Observation) bool { return o.Intensity.GT(math.LegacyZeroDec()) })
	if len(degrees) != 2 {
		t.Errorf("Expected 2 observations with intensity > 0, got %d", len(degrees))
	}
}

func TestFilterByNode(t *testing.T) {
	observations := []types.Observation{
		{Validator: "nodeA", Context: "A"},
		{Validator: "nodeB", Context: "B"},
		{Validator: "nodeA", Context: "C"},
	}

	nodeAObs := filterByNode(observations, "nodeA")
	if len(nodeAObs) != 2 {
		t.Errorf("Expected 2 observations from nodeA, got %d", len(nodeAObs))
	}
}
