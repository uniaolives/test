# tests/test_erl.py
import pytest
from papercoder_kernel.merkabah.self_node import SelfNode
from papercoder_kernel.cognition.erl import ExperientialLearning
from papercoder_kernel.cognition.utils import Memory, Environment

def test_erl_success_first_attempt():
    self_node = SelfNode()
    memory = Memory()
    # Mock environment to return high reward for y1
    class HighRewardEnv:
        def evaluate(self, y): return "Success", 0.9

    erl = ExperientialLearning(self_node, memory, HighRewardEnv(), threshold=0.5)
    result = erl.episode("test_task")

    assert result['r1'] == 0.9
    assert result['y2'] is None
    assert result['delta'] is None
    assert len(memory.retrieve()) == 0

def test_erl_refinement_cycle():
    self_node = SelfNode()
    memory = Memory()
    env = Environment() # Default env gives 0.3 for y1 and 0.9 for y2

    erl = ExperientialLearning(self_node, memory, env, threshold=0.5)
    result = erl.episode("test_task")

    assert result['r1'] == 0.3
    assert result['delta'] is not None
    assert result['r2'] == 0.9
    assert result['reward_reflect'] == 0.9
    assert len(memory.retrieve()) == 1
    assert memory.retrieve()[0] == result['delta']

def test_erl_failed_refinement():
    self_node = SelfNode()
    memory = Memory()
    # Mock environment to return low reward for both
    class LowRewardEnv:
        def evaluate(self, y): return "Fail", 0.2

    erl = ExperientialLearning(self_node, memory, LowRewardEnv(), threshold=0.5)
    result = erl.episode("test_task")

    assert result['r1'] == 0.2
    assert result['r2'] == 0.2
    assert result['reward_reflect'] == 0
    assert len(memory.retrieve()) == 0
