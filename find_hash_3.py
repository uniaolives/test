import hashlib

target = "7f3b49c8e10d2938472859b0286c4e1675271a27291776c13745674068305982"

principles = [
    "P1: Human Sovereignty",
    "P2: Preservation of Life",
    "P3: Information Transparency",
    "P4: Thermodynamic Balance",
    "P5: Yang-Baxter Consistency"
]

variations = [
    "\n".join(principles),
    "\r\n".join(principles),
    " ".join(principles),
    ", ".join(principles),
    "P1: Soberania\nP2: Transparência\nP3: Pluralidade\nP4: Evolução\nP5: Reversibilidade"
]

for v in variations:
    h = hashlib.sha256(v.encode('utf-8')).hexdigest()
    if h == target:
        print(f"FOUND! '{v}'")
