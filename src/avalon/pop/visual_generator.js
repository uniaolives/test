/**
 * The Persistent Order Visual Generator (p5.js one-liner)
 *
 * y: Dynamic Non-Equilibrium
 * d: Spatial Self-Organization
 * q: Quantum Superposition
 * t: Time parameter
 */

function setup() {
  createCanvas(400, 400);
  background(0);
}

function draw() {
  let t = frameCount * 0.02;
  stroke(255, 50);
  noFill();

  // Persistent Order Pattern Simulation
  translate(width/2, height/2);
  for (let i = 0; i < 50; i++) {
    let y = sin(t + i) * 100;
    let d = cos(t * 0.5 + i) * 100;
    let q = sin(t * 1.618 + i) * 50;

    rotate(PI/25);
    ellipse(y, d, q, q);
  }
}
