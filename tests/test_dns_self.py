
import asyncio
import pytest
from avalon.quantum.dns import QuantumDNSServer, QuantumDNSClient

@pytest.mark.asyncio
async def test():
    s = QuantumDNSServer()
    c = QuantumDNSClient(s)
    res = await c.query('qhttp://rabbithole.megaeth.com')
    print(res)
    assert res["status"] == "RESOLVED"
    assert res["entanglement_status"] == "SELF-ENTANGLED"
    print("Self-recognition test passed!")

if __name__ == "__main__":
    asyncio.run(test())
