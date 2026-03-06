import pytest
import asyncio
from gateway.app.blockchain.satoshi import SatoshiHypothesisVerifier, verify_satoshi_temporal

def test_satoshi_verifier():
    verifier = SatoshiHypothesisVerifier()
    # High squeezing case
    blocks = [
        {'nonce': 100, 'timestamp': 1000, 'difficulty': 1},
        {'nonce': 200, 'timestamp': 1001, 'difficulty': 1},
        {'nonce': 300, 'timestamp': 1002, 'difficulty': 1}
    ]
    sig = verifier.temporal_signature_entropy(blocks)
    assert sig['temporal_squeezing'] > 10.0
    assert sig['arkhe_score'] > 0.5

@pytest.mark.asyncio
async def test_satoshi_verification_loop():
    blocks = [{'nonce': i, 'timestamp': i*0.1, 'difficulty': 1} for i in range(100)]
    result = await verify_satoshi_temporal(blocks)
    assert result['verdict'] == 'TEMPORAL_SIGNATURE_DETECTED'
