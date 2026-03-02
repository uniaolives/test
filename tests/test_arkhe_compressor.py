import pytest
from metalanguage.arkhe_compressor import UniversalCodeHypergraph, ArkheCompressor

def test_python_parsing():
    hypergraph = UniversalCodeHypergraph()
    code = "[f(x) for x in lista if x > 0]"
    nodes, edges = hypergraph.parse_python(code)

    # Expected nodes: lista_node, filter_x, map_x
    assert len(nodes) == 3
    node_types = [n.node_type for n in nodes]
    assert "variable" in node_types
    assert "filter" in node_types
    assert "map" in node_types

    # Expected edges: lista -> filter, filter -> map
    assert len(edges) == 2

def test_haskell_parsing():
    hypergraph = UniversalCodeHypergraph()
    code = "map f (filter (>0) lista)"
    nodes, edges = hypergraph.parse_haskell(code)

    # Expected nodes: lista_node, filter_haskell, map_haskell
    assert len(nodes) == 3

    # Expected edges: lista -> filter, filter -> map
    assert len(edges) == 2

def test_pattern_equivalence():
    hypergraph = UniversalCodeHypergraph()

    python_code = "[f(x) for x in lista if x > 0]"
    hypergraph.add_code(python_code, 'python')

    haskell_code = "map f (filter (>0) lista)"
    hypergraph.add_code(haskell_code, 'haskell')

    equivalences = hypergraph.identify_pattern_equivalence()

    # We expect some equivalences based on node types (filter, map, etc.)
    assert len(equivalences) > 0

    found_filter = False
    for eq in equivalences:
        if eq['pattern'] == 'filter':
            assert 'python' in eq['languages']
            assert 'haskell' in eq['languages']
            found_filter = True

    assert found_filter

def test_transpilation():
    compressor = ArkheCompressor()
    python_code = "[f(x) for x in lista if x > 0]"

    result = compressor.compress_and_transpile(python_code, 'python', 'haskell')

    assert result['source_language'] == 'python'
    assert result['target_language'] == 'haskell'
    assert "map f (filter (>0) lista)" in result['target_code']
    assert result['compression_ratio'] == 3.2
