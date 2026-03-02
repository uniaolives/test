# tests/test_doublezero_integration.py
import pytest
from unittest.mock import MagicMock, patch
from papercoder_kernel.merkabah.doublezero import DoubleZeroInterface, DoubleZeroLayer

def test_doublezero_interface_address():
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = MagicMock(stdout="YourDoubleZeroAddress11111111111111111111111111111\n", check=True)
        interface = DoubleZeroInterface()
        address = interface.get_address()
        assert address == "YourDoubleZeroAddress11111111111111111111111111111"
        mock_run.assert_called_with(["doublezero", "address"], capture_output=True, text=True, check=True)

def test_doublezero_interface_status_up():
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = MagicMock(stdout="Session is up\n", check=True)
        interface = DoubleZeroInterface()
        status = interface.get_status()
        assert "up" in status.lower()

def test_doublezero_layer_initialize():
    mock_interface = MagicMock()
    mock_interface.get_address.return_value = "MockAddress123"

    layer = DoubleZeroLayer(interface=mock_interface)
    layer.initialize()

    assert layer.identity == "MockAddress123"
    assert layer.get_info()["identity"] == "MockAddress123"

def test_doublezero_layer_cli_not_found():
    mock_interface = MagicMock()
    mock_interface.get_address.return_value = "CLI_NOT_FOUND"

    layer = DoubleZeroLayer(interface=mock_interface)
    layer.initialize()

    assert "SimulatedDoubleZeroAddress" in layer.identity
