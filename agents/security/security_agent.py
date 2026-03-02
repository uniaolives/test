# agents/security/security_agent.py
from datetime import datetime
import asyncio
try:
    import nats
except ImportError:
    nats = None
try:
    from scapy.all import sniff, IP, TCP
except ImportError:
    sniff, IP, TCP = None, None, None
import psutil
import os
import sys

# Cosmopsychia integration
sys.path.append(os.getcwd())
from cosmos.network import WormholeNetwork
from cosmos.bridge import schumann_generator

class SecurityAgent:
    """Agente especializado em segurança da rede distribuída com lógica Cosmopsychia"""

    def __init__(self):
        self.threat_level = 0
        self.blocked_ips = set()
        self.audit_log = []
        self.wormhole_net = WormholeNetwork(node_count=64)
        self.resonance_freq = schumann_generator(n=1)

    async def monitor_network(self):
        if not sniff:
            print("⚠️ Scapy not installed. Network monitoring disabled.")
            return

        def packet_handler(pkt):
            if IP in pkt:
                src = pkt[IP].src
                curvature = self.wormhole_net.calculate_curvature(0, hash(src) % 64)
                if curvature < -2.0:
                    print(f"🛡️ High curvature detected from {src}")

                if TCP in pkt and pkt[TCP].flags == 'S':
                    self._detect_port_scan(src)
                if src in self.blocked_ips:
                    pass

        print(f"🔍 Starting network sniff at {self.resonance_freq}Hz resonance...")
        await asyncio.to_thread(sniff, prn=packet_handler, store=0, daemon=True)

    async def audit_access(self, user: str, resource: str, action: str):
        entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'user': user,
            'resource': resource,
            'action': action,
            'node': 'security_agent'
        }
        self.audit_log.append(entry)
        if self._detect_suspicious_pattern(user):
            await self._alert_brain(f"Suspicious activity: {user}")

    def _detect_port_scan(self, ip: str):
        self.threat_level += 0.1
        if self.threat_level > 1.0:
            self.blocked_ips.add(ip)

    def _detect_suspicious_pattern(self, user: str) -> bool:
        user_logs = [e for e in self.audit_log if e['user'] == user]
        recent = [e for e in user_logs
                 if (datetime.utcnow() - datetime.fromisoformat(e['timestamp'])).seconds < 60]
        return len(recent) > 10

    async def _alert_brain(self, message: str):
        if not nats:
            print(f"📡 [MOCK ALERT] {message}")
            return
        nats_url = os.environ.get("NATS_URL", "nats://localhost:4222")
        try:
            nc = await nats.connect(nats_url)
            await nc.publish("security.alerts", message.encode())
            await nc.close()
        except Exception as e:
            print(f"❌ Failed to alert brain: {e}")

async def main():
    agent = SecurityAgent()
    print("🛡️ Security Agent starting...")
    # await agent.monitor_network()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
