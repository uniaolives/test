import sys
import os
import numpy as np
import pytest

# Add ArkheOS src to path
sys.path.append(os.path.join(os.getcwd(), 'ArkheOS/src'))

from arkhe.interfaces.linux_eth_bridge import LinuxEthBridge

def test_linux_to_eth_handover():
    bridge = LinuxEthBridge()
    bridge.register_process(1001, "test_proc")
    bridge.register_contract("0xABC", "hash_abc")

    initial_satoshi = bridge.processes[1001].satoshi
    initial_balance = bridge.contracts["0xABC"].balance_satoshi

    success = bridge.linux_to_eth_call(1001, "0xABC", "doWork()", 10.0)

    # Check satoshi transfer
    assert bridge.processes[1001].satoshi == initial_satoshi - 10.0
    assert bridge.contracts["0xABC"].balance_satoshi == initial_balance + 9.0

    # Check coherence update
    if success:
        assert bridge.processes[1001].coherence > 0.98
    else:
        assert bridge.processes[1001].coherence < 0.98

def test_eth_to_linux_handover():
    bridge = LinuxEthBridge()
    bridge.register_process(2002, "event_handler")
    bridge.register_contract("0xDEF", "hash_def")

    initial_satoshi = bridge.processes[2002].satoshi

    success = bridge.eth_to_linux_event("0xDEF", "Alert", 2002)

    if success:
        assert bridge.processes[2002].satoshi == initial_satoshi + 1.0
        assert bridge.processes[2002].coherence > 0.98

def test_invalid_handover():
    bridge = LinuxEthBridge()
    # No registration
    assert bridge.linux_to_eth_call(999, "0xNON", "fail()", 1.0) is False
    assert bridge.eth_to_linux_event("0xNON", "Fail", 999) is False

if __name__ == "__main__":
    pytest.main([__file__])
