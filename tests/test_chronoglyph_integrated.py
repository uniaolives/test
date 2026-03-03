import sys
import os

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metalanguage.chronoglyph_runtime import demo_collapse
from metalanguage.chronoglyph_decoder import demo_85bit_chronoglyph
from metalanguage.chronoglyph_parser import SVGTopoParser, EXAMPLE_SVG

def test_chronoglyph_integrated():
    print("--- Testing Quantum-Symbolic Collapse Motor ---")
    universes = demo_collapse()
    assert len(universes) > 0
    print("✅ Motor demo successful\n")

    print("--- Testing 85-bit Decoder ---")
    decoder, results = demo_85bit_chronoglyph()
    assert len(decoder.graph.nodes) > 0
    assert len(results) > 0
    assert os.path.exists("85bit_program.svg")
    print("✅ 85-bit decoder demo successful\n")

    print("--- Testing SVG Topological Parser ---")
    parser = SVGTopoParser(EXAMPLE_SVG)
    parser.extract_circles()
    parser.build_topology()
    cg = parser.to_chronograph()

    assert len(parser.elements) > 0
    assert len(cg.nodes) > 0
    # The SVG has 2 circles for System 1, 1 for System 2, and 1 small point.
    # build_topology should find at least two systems.
    print(f"SVG Elements found: {len(parser.elements)}")
    print(f"ChronoGraph Nodes created: {len(cg.nodes)}")
    print("✅ SVG parser successful\n")

if __name__ == "__main__":
    try:
        test_chronoglyph_integrated()
        print("ALL CHRONOGLYPH INTEGRATED TESTS PASSED! 🌀✨")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
