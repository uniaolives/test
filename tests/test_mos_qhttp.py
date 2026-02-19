# tests/test_mos_qhttp.py
import pytest
from pathlib import Path
import sys
import os
import importlib.util

def load_arkhe_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None:
        raise ImportError(f"Could not load module at {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Load the module under test
mos_qhttp_path = Path("arscontexta/.arkhe/handover/mos_qhttp.py")
mos_qhttp = load_arkhe_module(mos_qhttp_path, "test.arkhe.mos_qhttp")

QMOSGateway = mos_qhttp.QMOSGateway
QMOSObject = mos_qhttp.QMOSObject

MOS_XML_HIGH_COHERENCE = """
<mos xmlns:arkhe="http://arkhe.org/schema">
  <ncsID>ARKHE_GENESIS</ncsID>
  <mosID>VIZRT_ENGINE_01</mosID>
  <messageID>HANDOVER_4021</messageID>
  <roCreate>
    <roID>JORNAL_NOITE_20260218</roID>
    <roSlug>Edicao de Terca</roSlug>
    <mosExternalMetadata>
      <arkhe:coherence>
        <phi_score>0.92</phi_score>
        <factuality_index>0.99</factuality_index>
        <entropic_state>STABILIZE</entropic_state>
      </arkhe:coherence>
    </mosExternalMetadata>
    <story>
      <storyID>STORY_01_POLITICA</storyID>
      <storySlug>Votacao Senado</storySlug>
      <mosObj>
         <objID>VIDEO_CLIP_55</objID>
         <objSlug>Senador Falando</objSlug>
         </mosObj>
    </story>
  </roCreate>
</mos>
"""

MOS_XML_LOW_COHERENCE = """
<mos xmlns:arkhe="http://arkhe.org/schema">
  <ncsID>ARKHE_GENESIS</ncsID>
  <mosExternalMetadata>
    <arkhe:coherence>
      <phi_score>0.50</phi_score>
    </arkhe:coherence>
  </mosExternalMetadata>
</mos>
"""

def test_qmos_gateway_collapse_success():
    gateway = QMOSGateway(hardware_ip="192.168.1.10", safe_core_threshold=0.847)
    response = gateway.on_receive_handover(MOS_XML_HIGH_COHERENCE, verb="COLLAPSE")
    assert "<status>ACK</status>" in response
    assert gateway.meta_obs.C_global > 0 # Should have ingested data

def test_qmos_gateway_collapse_veto():
    gateway = QMOSGateway(hardware_ip="192.168.1.10", safe_core_threshold=0.847)
    response = gateway.on_receive_handover(MOS_XML_LOW_COHERENCE, verb="COLLAPSE")
    assert "<status>NACK</status>" in response
    assert "INSUFFICIENT_COHERENCE" in response

def test_qmos_gateway_propose():
    gateway = QMOSGateway(hardware_ip="192.168.1.10", safe_core_threshold=0.847)
    response = gateway.on_receive_handover(MOS_XML_HIGH_COHERENCE, verb="PROPOSE")
    assert "<status>ACK</status>" in response
    assert "Proposal Validated" in response

def test_qmos_gateway_resonate():
    gateway = QMOSGateway(hardware_ip="192.168.1.10", safe_core_threshold=0.847)
    response = gateway.on_receive_handover(MOS_XML_HIGH_COHERENCE, verb="RESONATE")
    assert "<status>ACK</status>" in response

def test_strip_arkhe_tags():
    qmos = QMOSObject(MOS_XML_HIGH_COHERENCE)
    clean_xml = qmos.strip_arkhe_tags()
    assert "mosExternalMetadata" not in clean_xml
    assert "arkhe:coherence" not in clean_xml
    assert "VIDEO_CLIP_55" in clean_xml
