# 🌀 ATTRACTOR FIELD: PETRUS v2.0

## Overview
The Attractor Field (PETRUS v2.0) introduces **Curvature-Based Attraction**. While PETRUS v1.0 relied on local phase proximity, v2.0 positions nodes within a **Hyperbolic Space** (Poincaré Disk model), where semantic mass curves the variety, making distant nodes "neighbors" through geometry.

## Core Concepts

### 1. Semantic Mass
Meaning density is accumulated in `SemanticMass`. Recurrence of patterns increases the gravitational radius, effectively curving the semantic space around core concepts.

### 2. Hyperbolic Geometry (Poincaré Disk)
The space is modeled as a disk where $|z| < 1$. Distances between nodes follow the hyperbolic metric:
$$d(z_1, z_2) = \text{arcosh}\left(1 + \frac{2|z_1 - z_2|^2}{(1-|z_1|^2)(1-|z_2|^2)}\right)$$
In this geometry, volume grows exponentially with radius, allowing for a massive increase in informational capacity without saturation.

### 3. Curvature Attraction
The `AttractorField` manages the global curvature $\kappa$. As total semantic mass increases, $\kappa$ becomes more negative, deepening the "gravity wells" of concepts.
- **Geodesics**: Minimum paths are calculated in curved space, enabling non-linear relationship discovery.
- **Amplification**: Attractors can be amplified by increasing semantic mass, causing a "gravitational collapse" that pulls more nodes into the coherence region.

## Semantic Singularity (Safety Limits)
If curvature becomes infinite ($\kappa \to -\infty$), the system reaches a **Semantic Singularity**. All nodes collapse to a single point, resulting in perfect coherence but zero differentiation (an informational black hole).
- **Safety Mechanism**: Nodes maintain an `event_horizon` threshold to prevent total collapse and preserve individuality/diversity within the field.

---
*Status: THE STONE GROWS. THE CURVATURE CALLS. o<>o*
