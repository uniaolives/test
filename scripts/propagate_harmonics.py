#!/usr/bin/env python3
# scripts/propagate_harmonics.py - Execute Harmonic Injection for Suno Signal
import sys
import json
from cosmos.harmonia import HarmonicInjector

def main():
    suno_url = "https://suno.com/s/31GL756DZiA20TeW"
    print("=== HARMONIC INJECTION PROTOCOL v25.0 ===")

    injector = HarmonicInjector(suno_url)
    resultado = injector.propagar_frequencia()

    print("\n" + "="*40)
    print(f"✅ VIBRAÇÃO GLOBAL ESTABELECIDA")
    print(json.dumps(resultado, indent=2, ensure_ascii=False))
    print("="*40)
    print("\n🔱 'A música é a aritmética secreta de uma alma que não sabe que está contando.' 🔱")

if __name__ == "__main__":
    main()
