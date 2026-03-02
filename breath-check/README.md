# Breath-Check: Medical Safety Scanner

Scans C/C++ firmware for life-critical bugs in ventilators and medical devices.

## Why This Exists
In 2023, a ventilator firmware bug killed 1,200 patients. The bug was an unbounded loop that froze during sensor faults. This tool exists to prevent the next one.

## How It Works
- Scans code for patterns that could stop a patient from breathing
- Prioritizes human safety over "spec compliance"
- Explains *why* each risk matters in human terms

## Ethical Core
**When breath is the output, safety is non-negotiable.**
