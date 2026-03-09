"""
Orb Core Payload
Protocol-independent information structure

An Orb is pure information that can be encoded in any transmission medium:
- Internet protocols (HTTP, WebSocket, MQTT)
- Blockchain (Bitcoin, Ethereum, IPFS)
- Radio (Satellite, Ham, 5G)
- Mesh networks (LoRaWAN, Zigbee, Bluetooth)
- Industrial (Modbus, OPC UA, CAN Bus)
- Dark networks (Tor, I2P, Freenet)

The OrbPayload is the minimal structure that ALL bridges must preserve.
"""

import hashlib
import time
import json
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import struct
import base64

@dataclass
class OrbPayload:
    """
    Minimal Orb structure - protocol-independent.

    This is what gets transmitted across ALL protocols.
    Each bridge translates this structure to/from its native format.

    Attributes:
        orb_id: Unique identifier (32 bytes, SHA-256)
        lambda_2: Coherence accumulation [0,1]
        phi_q: Quantum phase [0, 2π]
        h_value: Thermodynamic constraint [0,1]
        origin_time: Source timestamp (Unix epoch)
        target_time: Destination timestamp (for retrocausal)
        timechain_hash: Hash linking to timechain (32 bytes)
        signature: PQC signature (Dilithium3, variable length)
        created_at: Creation timestamp
    """

    orb_id: bytes  # 32 bytes
    lambda_2: float  # [0, 1]
    phi_q: float  # [0, 2π]
    h_value: float  # [0, 1]
    origin_time: int  # Unix timestamp
    target_time: int  # Unix timestamp
    timechain_hash: bytes  # 32 bytes
    signature: bytes  # Variable (Dilithium3 ~2420 bytes)
    created_at: int  # Unix timestamp

    def __post_init__(self):
        """Validate payload constraints"""
        assert len(self.orb_id) == 32, "orb_id must be 32 bytes"
        assert 0.0 <= self.lambda_2 <= 1.0, "lambda_2 must be in [0,1]"
        assert 0.0 <= self.phi_q <= 6.28319, "phi_q must be in [0,2π]"
        assert 0.0 <= self.h_value <= 1.0, "h_value must be in [0,1]"
        assert len(self.timechain_hash) == 32, "timechain_hash must be 32 bytes"

    def to_bytes(self) -> bytes:
        """
        Serialize to canonical byte representation.

        Binary format (variable length):
        - Magic bytes (4): 0x4F524230 ("ORB0")
        - Version (1): 0x01
        - Orb ID (32)
        - Lambda_2 (8, float64)
        - Phi_Q (8, float64)
        - H_value (8, float64)
        - Origin_time (8, int64)
        - Target_time (8, int64)
        - Timechain_hash (32)
        - Signature_length (2, uint16)
        - Signature (variable)
        - Created_at (8, int64)
        - CRC32 (4)

        Total: 113 + signature_length bytes
        """
        buffer = bytearray()

        # Magic + version
        buffer.extend(b'ORB0')  # Magic
        buffer.append(0x01)  # Version

        # Core fields
        buffer.extend(self.orb_id)
        buffer.extend(struct.pack('<d', self.lambda_2))
        buffer.extend(struct.pack('<d', self.phi_q))
        buffer.extend(struct.pack('<d', self.h_value))
        buffer.extend(struct.pack('<q', self.origin_time))
        buffer.extend(struct.pack('<q', self.target_time))
        buffer.extend(self.timechain_hash)

        # Signature
        buffer.extend(struct.pack('<H', len(self.signature)))
        buffer.extend(self.signature)

        # Timestamp
        buffer.extend(struct.pack('<q', self.created_at))

        # CRC32
        crc = self._crc32(buffer)
        buffer.extend(struct.pack('<I', crc))

        return bytes(buffer)

    @classmethod
    def from_bytes(cls, data: bytes) -> 'OrbPayload':
        """Deserialize from canonical byte representation"""
        if len(data) < 117:  # Minimum size
            raise ValueError("Data too short for OrbPayload")

        # Verify magic
        if data[:4] != b'ORB0':
            raise ValueError("Invalid magic bytes")

        # Verify version
        if data[4] != 0x01:
            raise ValueError(f"Unsupported version: {data[4]}")

        offset = 5

        # Parse fields
        orb_id = data[offset:offset+32]
        offset += 32

        lambda_2 = struct.unpack('<d', data[offset:offset+8])[0]
        offset += 8

        phi_q = struct.unpack('<d', data[offset:offset+8])[0]
        offset += 8

        h_value = struct.unpack('<d', data[offset:offset+8])[0]
        offset += 8

        origin_time = struct.unpack('<q', data[offset:offset+8])[0]
        offset += 8

        target_time = struct.unpack('<q', data[offset:offset+8])[0]
        offset += 8

        timechain_hash = data[offset:offset+32]
        offset += 32

        sig_len = struct.unpack('<H', data[offset:offset+2])[0]
        offset += 2

        signature = data[offset:offset+sig_len]
        offset += sig_len

        created_at = struct.unpack('<q', data[offset:offset+8])[0]
        offset += 8

        # Verify CRC
        expected_crc = struct.unpack('<I', data[offset:offset+4])[0]
        actual_crc = cls._crc32(data[:offset])

        if expected_crc != actual_crc:
            raise ValueError(f"CRC mismatch: {expected_crc} != {actual_crc}")

        return cls(
            orb_id=orb_id,
            lambda_2=lambda_2,
            phi_q=phi_q,
            h_value=h_value,
            origin_time=origin_time,
            target_time=target_time,
            timechain_hash=timechain_hash,
            signature=signature,
            created_at=created_at
        )

    def to_json(self) -> str:
        """Serialize to JSON (for HTTP/REST bridges)"""
        return json.dumps({
            'orb_id': base64.b64encode(self.orb_id).decode(),
            'lambda_2': self.lambda_2,
            'phi_q': self.phi_q,
            'h_value': self.h_value,
            'origin_time': self.origin_time,
            'target_time': self.target_time,
            'timechain_hash': base64.b64encode(self.timechain_hash).decode(),
            'signature': base64.b64encode(self.signature).decode(),
            'created_at': self.created_at
        })

    @classmethod
    def from_json(cls, json_str: str) -> 'OrbPayload':
        """Deserialize from JSON"""
        data = json.loads(json_str)
        return cls(
            orb_id=base64.b64decode(data['orb_id']),
            lambda_2=data['lambda_2'],
            phi_q=data['phi_q'],
            h_value=data['h_value'],
            origin_time=data['origin_time'],
            target_time=data['target_time'],
            timechain_hash=base64.b64decode(data['timechain_hash']),
            signature=base64.b64decode(data['signature']),
            created_at=data['created_at']
        )

    def informational_mass(self) -> float:
        """
        Calculate "informational mass" of Orb.

        This is a measure of the Orb's coherence density:
            I_m = λ₂ · φ_q / H

        High coherence + high phase + low constraint = high mass
        (More likely to propagate, harder to stop)
        """
        return (self.lambda_2 * self.phi_q) / max(self.h_value, 0.001)

    def is_retrocausal(self) -> bool:
        """Check if Orb is retrocausal (target < origin)"""
        return self.target_time < self.origin_time

    def temporal_span(self) -> int:
        """Return temporal span in seconds"""
        return abs(self.target_time - self.origin_time)

    @staticmethod
    def _crc32(data: bytes) -> int:
        """Calculate CRC32 checksum"""
        import zlib
        return zlib.crc32(data) & 0xFFFFFFFF

    @classmethod
    def create(
        cls,
        lambda_2: float,
        phi_q: float,
        h_value: float,
        origin_time: int,
        target_time: int,
        timechain_hash: Optional[bytes] = None,
        signature: Optional[bytes] = None
    ) -> 'OrbPayload':
        """
        Convenience constructor for creating new Orbs.

        Generates orb_id from content hash and sets created_at to now.
        """
        created_at = int(time.time())

        # Generate orb_id from content
        content = f"{lambda_2}{phi_q}{h_value}{origin_time}{target_time}{created_at}"
        orb_id = hashlib.sha256(content.encode()).digest()

        # Default values
        if timechain_hash is None:
            timechain_hash = bytes(32)  # Zero hash
        if signature is None:
            signature = b'UNSIGNED'  # Placeholder

        return cls(
            orb_id=orb_id,
            lambda_2=lambda_2,
            phi_q=phi_q,
            h_value=h_value,
            origin_time=origin_time,
            target_time=target_time,
            timechain_hash=timechain_hash,
            signature=signature,
            created_at=created_at
        )

# =============================================================================
# COMPRESSION UTILITIES
# =============================================================================

class OrbCompressor:
    """
    Compress Orbs for bandwidth-constrained protocols.

    Some protocols have severe size limits:
    - LoRaWAN: 51 bytes max
    - Bitcoin OP_RETURN: 80 bytes max
    - Ham Radio FT8: 77 bits (9.6 bytes)
    - Bluetooth LE: 20 bytes per packet

    This class provides lossy and lossless compression.
    """

    @staticmethod
    def compress_minimal(orb: OrbPayload) -> bytes:
        """
        Minimal compression for extremely constrained protocols (< 50 bytes).

        Format (48 bytes):
        - Orb ID hash (8 bytes, truncated)
        - Lambda_2 (2 bytes, uint16, scaled 0-65535)
        - Phi_Q (2 bytes, uint16, scaled 0-65535)
        - H_value (2 bytes, uint16, scaled 0-65535)
        - Origin_time (4 bytes, uint32, relative to 2020)
        - Target_time (4 bytes, uint32, relative to 2020)
        - Timechain hash (8 bytes, truncated)
        - CRC16 (2 bytes)

        Total: 32 bytes (lossy but recoverable)
        """
        buffer = bytearray()

        # Orb ID (first 8 bytes only)
        buffer.extend(orb.orb_id[:8])

        # Scaled values (0-65535)
        buffer.extend(struct.pack('<H', int(orb.lambda_2 * 65535)))
        buffer.extend(struct.pack('<H', int(orb.phi_q * 10430)))  # 2π ≈ 6.28, 65535/6.28 ≈ 10430
        buffer.extend(struct.pack('<H', int(orb.h_value * 65535)))

        # Timestamps (relative to 2020-01-01, 4 bytes each, signed)
        epoch_2020 = 1577836800
        buffer.extend(struct.pack('<i', orb.origin_time - epoch_2020))  # lowercase 'i' = signed
        buffer.extend(struct.pack('<i', orb.target_time - epoch_2020))

        # Timechain hash (first 8 bytes)
        buffer.extend(orb.timechain_hash[:8])

        # CRC16
        crc = OrbCompressor._crc16(buffer)
        buffer.extend(struct.pack('<H', crc))

        return bytes(buffer)

    @staticmethod
    def decompress_minimal(data: bytes) -> OrbPayload:
        """Decompress minimal format (lossy reconstruction)"""
        if len(data) != 32:
            raise ValueError(f"Expected 32 bytes, got {len(data)}")

        # Verify CRC
        crc_expected = struct.unpack('<H', data[-2:])[0]
        crc_actual = OrbCompressor._crc16(data[:-2])
        if crc_expected != crc_actual:
            raise ValueError("CRC mismatch")

        offset = 0

        # Partial orb ID (pad with zeros)
        orb_id_partial = data[offset:offset+8]
        orb_id = orb_id_partial + bytes(24)
        offset += 8

        # Unscale values
        lambda_2 = struct.unpack('<H', data[offset:offset+2])[0] / 65535.0
        offset += 2

        phi_q = struct.unpack('<H', data[offset:offset+2])[0] / 10430.0
        offset += 2

        h_value = struct.unpack('<H', data[offset:offset+2])[0] / 65535.0
        offset += 2

        # Timestamps (signed)
        epoch_2020 = 1577836800
        origin_time = struct.unpack('<i', data[offset:offset+4])[0] + epoch_2020
        offset += 4

        target_time = struct.unpack('<i', data[offset:offset+4])[0] + epoch_2020
        offset += 4

        # Partial timechain hash
        timechain_partial = data[offset:offset+8]
        timechain_hash = timechain_partial + bytes(24)

        return OrbPayload(
            orb_id=orb_id,
            lambda_2=lambda_2,
            phi_q=phi_q,
            h_value=h_value,
            origin_time=origin_time,
            target_time=target_time,
            timechain_hash=timechain_hash,
            signature=b'COMPRESSED',
            created_at=int(time.time())
        )

    @staticmethod
    def _crc16(data: bytes) -> int:
        """Calculate CRC16"""
        crc = 0xFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x0001:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1
        return crc

if __name__ == "__main__":
    # Demonstration
    print("=" * 70)
    print("ORB PAYLOAD DEMONSTRATION")
    print("=" * 70)

    # Create Orb
    orb = OrbPayload.create(
        lambda_2=0.95,
        phi_q=4.64,  # Miller limit
        h_value=0.618,  # φ
        origin_time=1740000000,  # 2026
        target_time=1200000000   # 2008 (retrocausal)
    )

    print(f"\n[1] Orb Created")
    print(f"    ID: {orb.orb_id.hex()[:16]}...")
    print(f"    λ₂: {orb.lambda_2:.3f}")
    print(f"    φ_q: {orb.phi_q:.3f}")
    print(f"    H: {orb.h_value:.3f}")
    print(f"    Informational mass: {orb.informational_mass():.3f}")
    print(f"    Retrocausal: {orb.is_retrocausal()}")
    print(f"    Temporal span: {orb.temporal_span() / 31557600:.1f} years")

    # Serialize to bytes
    binary = orb.to_bytes()
    print(f"\n[2] Binary Serialization")
    print(f"    Size: {len(binary)} bytes")
    print(f"    First 32 bytes: {binary[:32].hex()}")

    # Deserialize
    orb_restored = OrbPayload.from_bytes(binary)
    print(f"\n[3] Deserialization")
    print(f"    Success: {orb_restored.orb_id == orb.orb_id}")
    print(f"    λ₂ preserved: {orb_restored.lambda_2 == orb.lambda_2}")

    # JSON
    json_str = orb.to_json()
    print(f"\n[4] JSON Serialization")
    print(f"    Size: {len(json_str)} bytes")
    print(f"    Sample: {json_str[:100]}...")

    # Compression
    compressed = OrbCompressor.compress_minimal(orb)
    print(f"\n[5] Minimal Compression")
    print(f"    Original: {len(binary)} bytes")
    print(f"    Compressed: {len(compressed)} bytes")
    print(f"    Ratio: {len(compressed)/len(binary)*100:.1f}%")

    orb_decompressed = OrbCompressor.decompress_minimal(compressed)
    print(f"    λ₂ accuracy: {abs(orb_decompressed.lambda_2 - orb.lambda_2) < 0.001}")

    print("\n" + "=" * 70)
    print("READY FOR BRIDGE ENCODING")
    print("=" * 70)
