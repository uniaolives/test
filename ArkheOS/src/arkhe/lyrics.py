"""
Lyrical Node Analysis: "Harmony and Chaos".
Maps poetic verses to Arkhe principles and Semi-Dirac physics.
"""

from typing import Dict, List

class LyricalAnalyzer:
    def __init__(self, poem_text: str):
        self.poem = poem_text
        self.mappings = {
            "Mirrored lines": "Symmetry / Massive Direction (C)",
            "Chaos is dominant": "Fluctuation / Massless Direction (F)",
            "Order and chaos play a strange romance": "Dirac Point (C + F = 1)",
            "Shattered light": "Handover / Recombination",
            "Dazzling maze": "Hypergraph Topology"
        }

    def analyze_structure(self) -> Dict[str, str]:
        results = {}
        for fragment, principle in self.mappings.items():
            if fragment.lower() in self.poem.lower():
                results[fragment] = principle
        return results

def get_harmony_chaos_poem() -> str:
    return """
    Mirrored lines evenly span,
    raising waves follow the orderly plan.
    A feast of curved spiraling shells,
    our hearts adore the elegant grace.
    Yet beauty stands where senses stray,
    a flux of arcs gently meet.
    But chaos is dominant like a vast sea,
    that opens the doors to sets us free.
    Order and chaos play a strange romance,
    game of strict rules or a whimsical chance.
    Broken forms turn into art,
    for the shattered light to find a new start.
    In harmony we live soundly asleep,
    for the unknown we ought to dive deep.
    """

if __name__ == "__main__":
    poem = get_harmony_chaos_poem()
    analyzer = LyricalAnalyzer(poem)
    analysis = analyzer.analyze_structure()
    print("--- Poetic-Physical Analysis ---")
    for fragment, result in analysis.items():
        print(f"Verse Fragment: '{fragment}' -> Principle: {result}")
