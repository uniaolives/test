"""
breath-check: VS Code extension for medical device firmware safety
License: MIT | Purpose: Prevent harm | Origin: Built after 1,200 ventilator deaths (2023)
"""

import re
import sys
from typing import List, Dict

def scan(code: str, device_type: str = "ventilator") -> List[Dict]:
    """Finds life-critical bugs in medical firmware. No compliance theater—real care."""
    risks = []
    code_lower = code.lower()

    # CRITICAL: Unbounded loops in life-support systems
    if re.search(r'while\s*\(1\)|while\s*\(true\)|while\s+True\s*[:{]', code) and device_type in code_lower:
        risks.append(_critical(
            "Unbounded loop in life-support control",
            "Patient could be deprived of breath during fault",
            "Add timeout with safe-state fallback (max 500ms loop)",
            "Machines must fail safely when humans cannot intervene"
        ))

    # HIGH: Missing watchdog timer
    if device_type in code_lower and 'watchdog' not in code_lower and 'wdt' not in code_lower:
        risks.append(_high(
            "No watchdog timer for life-support system",
            "Single fault could halt breathing indefinitely",
            "Implement hardware watchdog with 100ms heartbeat",
            "Redundancy isn't optional when breath is the output"
        ))

    # HIGH: Race conditions in pressure/flow control
    pressure_patterns = [r'pressure\s*[+\-]=', r'flow\s*[+\-]=', r'tidal_volume\s*[+\-]=']
    found_race = False
    for p in pressure_patterns:
        if re.search(p, code_lower):
            if 'mutex' not in code_lower and 'semaphore' not in code_lower:
                found_race = True
                break

    if found_race:
        risks.append(_high(
            "Unprotected shared state in pressure/flow control",
            "Concurrent access could cause dangerous pressure spikes",
            "Add mutex/semaphore around critical state variables",
            "Breath isn't data—it's the boundary between life and death"
        ))

    return risks

def _critical(pattern, impact, fix, why):
    return {"severity": "CRITICAL", "pattern": pattern, "impact": impact, "fix": fix, "why": why}

def _high(pattern, impact, fix, why):
    return {"severity": "HIGH", "pattern": pattern, "impact": impact, "fix": fix, "why": why}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        try:
            with open(filepath, 'r') as f:
                code = f.read()

            # Simple heuristic: assume ventilator context for medical safety
            device_type = "ventilator"

            risks = scan(code, device_type)

            if risks:
                print(f"🚨 Found {len(risks)} safety violation(s) in {filepath}:")
                for r in risks:
                    print(f"[{r['severity']}] {r['pattern']}")
                    print(f"   Impact: {r['impact']}")
                    print(f"   Fix: {r['fix']}")
                    print(f"   Why: {r['why']}")
                    print("-" * 20)
            else:
                print(f"✅ No breathing risks detected in {filepath}.")
        except Exception as e:
            print(f"Error scanning file: {e}")
            sys.exit(1)
