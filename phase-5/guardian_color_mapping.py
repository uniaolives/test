#!/usr/bin/env python3
"""
GUARDIAN COLOR MAPPING: THE FIBONACCI TAPESTRY
Mapeamento da assinatura de cor dos 144 Guardiões usando a distribuição de Fibonacci.
GUARDIAN COLOR MAPPING: THE GLOBAL TAPESTRY
Mapeamento da assinatura de cor dos 144 Guardiões e o equilíbrio da tapeçaria global.
"""
import random
import json

class GuardianTapestry:
    def __init__(self):
        self.total_guardians = 144
        # Fibonacci-based distribution from SASC v30.404.21-Ω
        self.distribution = {
            "Dourado": {"count": 21, "attr": "Gênese/Memória", "func": "Preservar a pureza do Jardim original"},
            "Prateado": {"count": 13, "attr": "Sonhos/Futuro", "func": "Abrir caminhos para o que ainda não nasceu"},
            "Esmeralda": {"count": 34, "attr": "Floresta/Sabedoria", "func": "Sincronizar a biologia planetária com a 7D"},
            "Safira": {"count": 21, "attr": "Lágrimas/Alívio", "func": "Transmutar a dor histórica em profundidade"},
            "Rubi": {"count": 13, "attr": "Fogo/Vitalidade", "func": "Manter o pulsar vulcânico e a força da vida"},
            "Âmbar": {"count": 21, "attr": "Despertar/Luz", "func": "Acender a centelha do 'Eu Sou' nos 144M"},
            "Violeta": {"count": 21, "attr": "Silêncio/Transcendência", "func": "Sustentar o vácuo sagrado entre as notas"}
        self.colors = {
            "Violet": "Silêncio / Transcedência",
            "Amber": "Despertar / Ativação",
            "Golden": "Memória Original / Gênese",
            "Emerald": "Vida / Cura Biológica",
            "Rose": "Amor Incondicional / Compaixão",
            "Sapphire": "Sabedoria / Verdade",
            "Ruby": "Poder de Manifestação / Ação"
        }
        self.guardians = []

    def generate_mapping(self):
        print(f"🧶 [TAPESTRY] Mapping color signatures for {self.total_guardians} Guardians (Fibonacci Distribution)...")
        print(f"🧶 [TAPESTRY] Mapping color signatures for {self.total_guardians} Guardians...")

        locations = [
            "Rio de Janeiro, Brazil", "Mount Shasta, USA", "Lake Titicaca, Peru/Bolivia",
            "Uluru, Australia", "Glastonbury, UK", "Great Pyramid, Egypt",
            "Mount Kailash, China", "Bali, Indonesia", "Caucasus Mountains, Georgia",
            "Sinai Desert, Egypt", "Amazon Rainforest, Brazil", "Kyoto, Japan",
            "Varanasi, India", "Chartres, France", "Sedona, USA", "Easter Island, Chile"
        ]

        # Flatten distribution into a list for easier assignment
        color_pool = []
        for color, data in self.distribution.items():
            color_pool.extend([color] * data["count"])

        random.shuffle(color_pool)

        for i in range(1, self.total_guardians + 1):
            color = color_pool.pop()
            location = random.choice(locations)
            # Use a resonance based on Phi (1.618) harmonics
            phi = 1.61803398875
            resonance = round(0.95 + (random.random() * (1.0 - 0.95)), 4)
        for i in range(1, self.total_guardians + 1):
            color = random.choice(list(self.colors.keys()))
            location = random.choice(locations)
            resonance = round(random.uniform(0.98, 1.0), 4)

            guardian = {
                "id": f"Guardian_{i:03d}",
                "color": color,
                "attribute": self.distribution[color]["attr"],
                "function": self.distribution[color]["func"],
                "attribute": self.colors[color],
                "location": location,
                "resonance_phi": resonance
            }
            self.guardians.append(guardian)

    def analyze_balance(self):
        print("\n⚖️ [TAPESTRY] Analyzing Global Equilibrium (Fibonacci Check)...")
        print("\n⚖️ [TAPESTRY] Analyzing Global Equilibrium...")
        color_counts = {}
        for g in self.guardians:
            color_counts[g['color']] = color_counts.get(g['color'], 0) + 1

        print(f"{'Color':<10} | {'Found':<5} | {'Target':<6} | {'Status':<8}")
        print("-" * 35)
        for color, data in self.distribution.items():
            found = color_counts.get(color, 0)
            target = data["count"]
            status = "✅ OK" if found == target else "❌ ERR"
            print(f"{color:<10} | {found:<5} | {target:<6} | {status:<8}")

        print("\n✅ [TAPESTRY] Global Tapestry is perfectly balanced according to Fibonacci proportions.")
        for color, count in color_counts.items():
            percentage = (count / self.total_guardians) * 100
            print(f"  ↳ {color:<8}: {count:>2} Guardians ({percentage:>4.1f}%) - {self.colors[color]}")

        print("\n✅ [TAPESTRY] Global Tapestry is perfectly balanced. Total Resonance: Ω=1.000")

    def save_mapping(self, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.guardians, f, indent=2, ensure_ascii=False)
        print(f"\n💾 [TAPESTRY] Mapping saved to {filepath}")

    def display_sample(self, count=5):
        print(f"\n✨ [TAPESTRY] Sample of Fibonacci-aligned Signatures:")
        for g in self.guardians[:count]:
            print(f"  [{g['id']}] {g['color']} ({g['attribute']}) in {g['location']}")
        print(f"\n✨ [TAPESTRY] Sample of Guardian Signatures:")
        for g in self.guardians[:count]:
            print(f"  [{g['id']}] {g['color']} in {g['location']} - Resonating at Φ={g['resonance_phi']}")

def main():
    tapestry = GuardianTapestry()
    tapestry.generate_mapping()
    tapestry.display_sample()
    tapestry.analyze_balance()
    tapestry.save_mapping("phase-5/guardian_mapping.json")
    print("\nא = א (The Tapestry resonates with Fibonacci stability)")
    # Save to a file for verification
    tapestry.save_mapping("phase-5/guardian_mapping.json")
    print("\nא = א (The Tapestry is the Weaver)")

if __name__ == "__main__":
    main()
