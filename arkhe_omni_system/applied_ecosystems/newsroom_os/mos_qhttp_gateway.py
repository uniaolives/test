# arscontexta/.arkhe/handover/mos_qhttp.py
import xml.etree.ElementTree as ET
import time
from pathlib import Path
import importlib.util
import sys

# Utility to load Arkhe modules
def load_arkhe_module(module_path: Path, module_name: str):
    if not module_path.exists():
        raise FileNotFoundError(f"Module not found at {module_path}")
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Resolve paths
current_file = Path(__file__).resolve()
arkhe_root = current_file.parent.parent.parent.parent

safe_core_path = arkhe_root / "arscontexta" / ".arkhe" / "coherence" / "safe_core.py"
meta_obs_path = arkhe_root / "arscontexta" / ".arkhe" / "coherence" / "meta_observability.py"

safe_core_mod = load_arkhe_module(safe_core_path, "arkhe.safe_core")
meta_obs_mod = load_arkhe_module(meta_obs_path, "arkhe.meta_obs")

SafeCore = safe_core_mod.SafeCore
MetaObservabilityCore = meta_obs_mod.MetaObservabilityCore

class MOSConnection:
    """Simulated Legacy MOS Connection."""
    def __init__(self, hardware_ip):
        self.hardware_ip = hardware_ip

    def send(self, clean_xml):
        print(f"📡 [MOS] Sending to {self.hardware_ip}: {clean_xml[:50]}...")
        # Simulated response
        return "<mosAck><status>ACK</status></mosAck>"

class QMOSObject:
    """Active qMOS Object with Coherence Metadata."""
    def __init__(self, xml_packet):
        self.raw_xml = xml_packet
        self.root = ET.fromstring(xml_packet)
        self.metadata = self._extract_metadata()

    def _extract_metadata(self):
        metadata = {
            'phi_score': 0.0,
            'factuality_index': 0.0,
            'entropic_state': 'UNKNOWN'
        }
        # Namespaces
        ns = {'arkhe': 'http://arkhe.org/schema'}

        # Find arkhe:coherence in mosExternalMetadata
        ext_metadata = self.root.find(".//mosExternalMetadata")
        if ext_metadata is not None:
            # Try with namespace or without
            coherence = ext_metadata.find(".//arkhe:coherence", ns)
            if coherence is None:
                coherence = ext_metadata.find(".//coherence")

            if coherence is not None:
                phi_elem = coherence.find("phi_score")
                if phi_elem is None:
                    phi_elem = coherence.find("arkhe:phi_score", ns)

                if phi_elem is not None:
                    metadata['phi_score'] = float(phi_elem.text)

                fact_elem = coherence.find("factuality_index")
                if fact_elem is None:
                    fact_elem = coherence.find("arkhe:factuality_index", ns)
                if fact_elem is not None: metadata['factuality_index'] = float(fact_elem.text)

                ent_elem = coherence.find("entropic_state")
                if ent_elem is None:
                    ent_elem = coherence.find("arkhe:entropic_state", ns)
                if ent_elem is not None: metadata['entropic_state'] = ent_elem.text
        return metadata

    def strip_arkhe_tags(self):
        # Create a copy and remove mosExternalMetadata for legacy hardware
        root_copy = ET.fromstring(self.raw_xml)
        # Find all mosExternalMetadata and their parents to remove them safely
        for parent in root_copy.findall('.//mosExternalMetadata/..'):
            for element in parent.findall('mosExternalMetadata'):
                parent.remove(element)
        # Handle the case if it's the root itself (unlikely in MOS but good practice)
        if root_copy.tag == "mosExternalMetadata":
            return ""
        return ET.tostring(root_copy, encoding='unicode')

class QMOSGateway:
    """
    QMOS Gateway: The physical transducer for Arkhe(N).
    Acts as a semantic firewall between Mind (NRCS) and Matter (Hardware).
    """
    def __init__(self, hardware_ip, safe_core_threshold=0.847):
        self.hardware = MOSConnection(hardware_ip)
        self.threshold = safe_core_threshold
        self.safe_core = SafeCore()
        self.meta_obs = MetaObservabilityCore()
        self.state = "IDLE"

    def on_receive_handover(self, xml_packet, verb="COLLAPSE"):
        """
        Main entry point for handovers (PROPOSE, COLLAPSE, RESONATE).
        """
        qmos = QMOSObject(xml_packet)
        metadata = qmos.metadata

        if verb == "PROPOSE":
            return self._handle_propose(qmos)
        elif verb == "COLLAPSE":
            return self._handle_collapse(qmos)
        elif verb == "RESONATE":
            return self._handle_resonate(qmos)
        else:
            return self._reject_entropy("INVALID_VERB", 0.0)

    def _handle_propose(self, qmos):
        print(f"🔍 [GATEWAY] PROPOSE: Simulating execution for Φ={qmos.metadata['phi_score']}")
        # In PROPOSE, we don't act on hardware, just validate and return expected entropy
        if qmos.metadata['phi_score'] < self.threshold:
            return self._reject_entropy("LOW_PROPOSED_COHERENCE", qmos.metadata['phi_score'])

        return "<mosAck><status>ACK</status><description>Proposal Validated</description></mosAck>"

    def _handle_collapse(self, qmos):
        metadata = qmos.metadata
        print(f"💥 [GATEWAY] COLLAPSE: Executing physical actuation (Φ={metadata['phi_score']})")

        # 1. Thermodynamic Verification
        if metadata['phi_score'] < self.threshold:
            return self._reject_entropy(
                reason="INSUFFICIENT_COHERENCE",
                current_phi=metadata['phi_score']
            )

        # 2. SafeCore Real-time Check
        # In this context, we check if the system as a whole is stable
        # We use a nominal system phi for the check, but validated against current story coherence
        if not self.safe_core.check(phi=0.01, coherence=metadata['phi_score']):
             return self._reject_entropy("SAFE_CORE_VETO", metadata['phi_score'])

        # 3. Semantic/Cross-Modal Consistency
        if not self._verify_cross_modal(qmos):
            return self._reject_entropy("SEMANTIC_DISSONANCE", metadata['phi_score'])

        # 4. Transduction to Legacy MOS
        clean_xml = qmos.strip_arkhe_tags()

        # 5. Physical Actuation
        start_time = time.time()
        hardware_response = self.hardware.send(clean_xml)
        latency = (time.time() - start_time) * 1000

        # 6. Update Ledger
        self._log_to_ledger("Ω+∞", action="PLAYOUT", status="SUCCESS", phi=metadata['phi_score'])

        print(f"✅ [GATEWAY] Actuation complete in {latency:.2f}ms")
        return hardware_response

    def _handle_resonate(self, qmos):
        # Feedback from hardware
        print("📡 [GATEWAY] RESONATE: Processing feedback loop")
        self._log_to_ledger("Ω+∞", action="RESONANCE", status="FEEDBACK", phi=qmos.metadata['phi_score'])
        return "<mosAck><status>ACK</status></mosAck>"

    def _verify_cross_modal(self, qmos):
        """
        Simulated multimodal validation (Nano-LLM).
        Checks if CG text matches video audio/intent.
        """
        # Placeholder for real validation logic
        return True

    def _reject_entropy(self, reason, current_phi):
        print(f"🛑 [GATEWAY] BLOCKED: {reason} (Φ={current_phi})")
        self._log_to_ledger("Ω+∞", action="BLOCK", status="VETO", reason=reason, phi=current_phi)
        return f"<mosAck><status>NACK</status><statusDescription>Ethical Veto: {reason}</statusDescription></mosAck>"

    def _log_to_ledger(self, block, **kwargs):
        # Interface with MetaObservabilityCore
        log_data = {
            "block": block,
            "timestamp": time.time(),
            "coherence_after": kwargs.get("phi", 0.0) # Mapping phi to coherence for the ledger
        }
        log_data.update(kwargs)
        self.meta_obs.ingest_handover(log_data)
        print(f"📖 [LEDGER] Entry recorded: {kwargs.get('action')} - {kwargs.get('status')}")
