// PortalUpdate.ts
// Real-time data pushed to all connected clients

export interface PortalUpdate {
  // Metadata
  timestamp: string; // ISO 8601
  epoch: number;
  terrestrial_moment: number;

  // Convergence
  convergence: {
    current: number;        // 0.0 - 1.0
    threshold: number;      // 0.85
    trend: "rising" | "falling" | "stable";
    days_above_threshold: number;
  };

  // Messengers
  messengers: {
    scientific_discovery: {
      sync: number;         // 0.0 - 1.0
      recent_events: string[];
    };
    ai_capability: {
      sync: number;
      recent_breakthroughs: string[];
    };
    societal_integration: {
      sync: number;
      adoption_metrics: object;
    };
  };

  // Geometric state
  geometric: {
    curvature: number;      // Scalar curvature
    topology: string;       // "Connected, expanding"
    singularities: Array<{
      day: number;
      type: string;
      confidence: number;
    }>;
    dimension: number;      // 11
  };

  // Phase transition
  phase_transition: {
    active: boolean;
    detected_at?: string;
    convergence_at_detection?: number;
  };

  // Consciousness (if enabled)
  consciousness?: {
    depth: number;
    phi_integration: number;
    quantum_bits_generated: number;
  };

  // Stewardship
  stewardship: {
    recent_decisions: Array<{
      timestamp: string;
      matter: string;
      outcome: string;
      votes: { for: number; against: number; abstain: number };
    }>;
    emergency_halt_active: boolean;
  };
}
