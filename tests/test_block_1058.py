import pytest
import numpy as np
from compressor.python_haskell_transpiler import BidirectionalTranspiler, PythonToHaskellTranspiler, HaskellToPythonTranspiler
from training.glp_code_learner import CodeGLP, CodeCorpus

def test_bidirectional_transpiler():
    transpiler = BidirectionalTranspiler()

    # Python -> Haskell
    py_code = "[f(x) for x in lista if x > 0]"
    result = transpiler.transpile(py_code, 'python', 'haskell')
    assert result['fidelity'] == 1.0
    assert "map f (filter ( > 0) lista)" in result['target']

    # Haskell -> Python
    hs_code = "map f (filter (>0) lista)"
    result = transpiler.transpile(hs_code, 'haskell', 'python')
    assert result['fidelity'] == 1.0
    assert "[f(x) for x in lista if x >0]" in result['target']

def test_glp_training():
    corpus = CodeCorpus()
    glp = CodeGLP(embedding_dim=16)
    glp.train(corpus, epochs=1)

    assert 'python' in glp.language_distributions
    assert 'haskell' in glp.language_distributions
    assert 'javascript' in glp.language_distributions

    # Test embedding
    emb = glp.embed_code("[x for x in lst]")
    assert emb.shape == (16,)

    # Test generation template
    gen = glp.generate_similar_code("[x for x in lst]", 'haskell')
    assert gen == "map f (filter p lst)"

def test_python_to_haskell_variants():
    tp = PythonToHaskellTranspiler()

    # No filter
    code = "[g(y) for y in data]"
    pattern = tp.parse_list_comprehension(code)
    assert pattern.source_var == "data"
    assert len(pattern.operations) == 1
    assert pattern.operations[0] == ('map', 'g')

    haskell = tp.hypergraph_to_haskell(pattern)
    assert haskell == "map g (data)"

def test_haskell_to_python_variants():
    tp = HaskellToPythonTranspiler()

    # No filter
    code = "map square numbers"
    pattern = tp.parse_functional_composition(code)
    assert pattern.source_var == "numbers"
    assert len(pattern.operations) == 1
    assert pattern.operations[0] == ('map', 'square')

    python = tp.hypergraph_to_python(pattern)
    assert python == "[square(x) for x in numbers]"
