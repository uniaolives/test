import hashlib

target = "7f3b49c8e10d2938472859b0286c4e1675271a27291776c13745674068305982"
base = "ARKHE_PROTOCOL_OMEGA_215::CONSTITUTION_V1::P1_SOVEREIGNTY::P2_LIFE::P3_TRANSPARENCY::P4_THERMODYNAMICS::P5_CAUSALITY"

variations = [
    base,
    base + "\n",
    base + "\r\n",
    base.lower(),
    base.upper(),
    "P1_SOVEREIGNTY::P2_LIFE::P3_TRANSPARENCY::P4_THERMODYNAMICS::P5_CAUSALITY",
    "CONSTITUTION_V1::P1_SOVEREIGNTY::P2_LIFE::P3_TRANSPARENCY::P4_THERMODYNAMICS::P5_CAUSALITY",
]

for v in variations:
    h = hashlib.sha256(v.encode('utf-8')).hexdigest()
    if h == target:
        print(f"FOUND! '{v}'")
    else:
        print(f"Tried '{v[:20]}...', got {h[:10]}")
