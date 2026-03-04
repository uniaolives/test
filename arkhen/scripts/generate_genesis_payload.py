import hashlib

# Canonical string representing the Arkhe(n) Constitution Principles P1-P5
# Defined as the "Ethical Genome" of Arkhe(n)
constitution_text = "ARKHE_PROTOCOL_OMEGA_215::CONSTITUTION_V1::P1_SOVEREIGNTY::P2_LIFE::P3_TRANSPARENCY::P4_THERMODYNAMICS::P5_CAUSALITY"

# Hash calculation (SHA-256)
sha256_hash = hashlib.sha256(constitution_text.encode('utf-8')).hexdigest()

# Bitcoin OP_RETURN Payload (6a = OP_RETURN, 20 = 32 bytes of data)
payload = "6a20" + sha256_hash

print(f"--- ARKHE(N) GENESIS RITUAL ---")
print(f"Canonical Constitution Text: {constitution_text}")
print(f"SHA-256 Hash: {sha256_hash}")
print(f"Genesis Ritual OP_RETURN Payload: {payload}")
print(f"-------------------------------")
