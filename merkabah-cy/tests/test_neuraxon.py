import pytest
import time
import sys
import os

# Add the src path to sys.path directly
sys.path.insert(0, os.path.join(os.getcwd(), "merkabah-cy/src/python"))

# Import directly from the module file to avoid package-level imports that might fail
import importlib.util
spec = importlib.util.spec_from_file_location("neuraxon", "merkabah-cy/src/python/merkabah/agi/neuraxon.py")
neuraxon = importlib.util.module_from_spec(spec)
spec.loader.exec_module(neuraxon)

def test_trinary_handover():
    assert neuraxon.trinary_handover(0.8) == 1
    assert neuraxon.trinary_handover(-0.8) == -1
    assert neuraxon.trinary_handover(0.1) == 0
    assert neuraxon.trinary_handover(0.5) == 0
    assert neuraxon.trinary_handover(-0.5) == 0

def test_neuraxon_node():
    node = neuraxon.NeuraxonNode(node_id=1)
    assert node.output_state(0.9) == 1
    assert node.output_state(-0.9) == -1
    assert node.output_state(0.2) == 0

def test_structural_plasticity():
    graph = neuraxon.SmallWorldGraph(n_nodes=10)
    plasticity = neuraxon.StructuralPlasticity(graph)

    # Test Potentiation
    graph.silent_synapses.add((1, 2))
    plasticity.potentiate(1, 2, correlation=0.9)
    assert (1, 2) in graph.active_edges
    assert graph.edges[(1, 2)] == 0.9
    assert len(plasticity.ledger) == 1
    assert plasticity.ledger[0].type == "synapse_formation"

def test_structural_pruning():
    graph = neuraxon.SmallWorldGraph(n_nodes=10)
    plasticity = neuraxon.StructuralPlasticity(graph)

    graph.activate_edge(1, 2, weight=0.5)
    # Simulate inactivity
    current_time = time.time()
    plasticity.prune(1, 2, inactivity_duration=10.0, last_activation=current_time - 20.0)
    assert (1, 2) not in graph.active_edges
    assert len(plasticity.ledger) == 1
    assert plasticity.ledger[0].type == "synapse_elimination"
