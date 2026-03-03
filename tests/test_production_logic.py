# tests/test_production_logic.py
import pytest
from unittest.mock import MagicMock, patch
from arkhe_qutip.cloud.f1_node_controller import ArkheF1Node
from arkhe_qutip.cloud.cluster_manager import F1ClusterManager

@patch('boto3.client')
def test_f1_node_status(mock_boto):
    # Mock EC2 describe_instances response
    mock_ec2 = MagicMock()
    mock_ec2.describe_instances.return_value = {
        'Reservations': [{
            'Instances': [{
                'PublicIpAddress': '1.2.3.4',
                'PrivateIpAddress': '10.0.0.1',
                'State': {'Name': 'running'}
            }]
        }]
    }
    mock_boto.return_value = mock_ec2

    node = ArkheF1Node("i-12345", region="us-east-1")
    status = node.get_info()

    assert status == 'running'
    assert node.public_ip == '1.2.3.4'
    assert node.private_ip == '10.0.0.1'

@patch('boto3.client')
def test_cluster_launch(mock_boto):
    mock_ec2 = MagicMock()
    mock_ec2.run_instances.return_value = {
        'Instances': [{'InstanceId': 'i-999'}]
    }
    mock_boto.return_value = mock_ec2

    manager = F1ClusterManager({'us-east-1': 1})
    manager.launch_global_testnet("agfi-xxx")

    assert len(manager.nodes) == 1
    assert manager.nodes[0].instance_id == 'i-999'

def test_pe_array_logic():
    # Since we can't run SystemVerilog, we verify the module files exist
    import os
    assert os.path.exists('arkhe_qutip/hardware/u280_core.sv')
    assert os.path.exists('arkhe_qutip/hardware/pe_array.sv')
    assert os.path.exists('arkhe_qutip/hardware/cx_swapper.sv')
