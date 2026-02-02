# agents/security/security_agent.py
from datetime import datetime
import asyncio
import nats
try:
    from scapy.all import sniff, IP, TCP
except ImportError:
    sniff, IP, TCP = None, None, None
import psutil
import os

class SecurityAgent:
    """Agente especializado em segurança da rede distribuída"""

    def __init__(self):
        self.threat_level = 0
        self.blocked_ips = set()
        self.audit_log = []

    async def monitor_network(self):
        """Monitora tráfego de rede em tempo real.

        NOTE: This function requires root/administrative privileges or NET_ADMIN
        capability to perform packet sniffing.
        """
        if not sniff:
            print("⚠️ Scapy not installed. Network monitoring disabled.")
            return

        def packet_handler(pkt):
            if IP in pkt:
                src = pkt[IP].src
                dst = pkt[IP].dst

                # Detectar scans de porta
                if TCP in pkt and pkt[TCP].flags == 'S':
                    self._detect_port_scan(src)

                # Verificar contra lista de bloqueio
                if src in self.blocked_ips:
                    self._drop_packet(pkt)

        # Iniciar sniffing (requer privilégios)
        print("🔍 Starting network sniff...")
        await asyncio.to_thread(sniff, prn=packet_handler, store=0, daemon=True)

    async def audit_access(self, user: str, resource: str, action: str):
        """Audita acesso a recursos"""
        entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'user': user,
            'resource': resource,
            'action': action,
            'node': 'security_agent'
        }

        self.audit_log.append(entry)

        # Alertar se padrão suspeito
        if self._detect_suspicious_pattern(user):
            await self._alert_brain(f"Suspicious activity: {user}")

    def _detect_port_scan(self, ip: str):
        """Detecta tentativas de port scan"""
        print(f"🕵️ Potential port scan detected from: {ip}")
        self.threat_level += 0.1
        if self.threat_level > 1.0:
            self.blocked_ips.add(ip)

    def _detect_suspicious_pattern(self, user: str) -> bool:
        """Detecta padrões de acesso suspeitos"""
        # Análise de logs
        user_logs = [e for e in self.audit_log if e['user'] == user]

        # Muitos acessos em curto tempo
        recent = [e for e in user_logs
                 if (datetime.utcnow() - datetime.fromisoformat(e['timestamp'])).seconds < 60]

        return len(recent) > 10  # Limiar arbitrário

    async def _alert_brain(self, message: str):
        """Envia alerta ao Brain central"""
        nats_url = os.environ.get("NATS_URL", "nats://localhost:4222")
        try:
            nc = await nats.connect(nats_url)
            await nc.publish("security.alerts", message.encode())
            await nc.close()
        except Exception as e:
            print(f"❌ Failed to alert brain: {e}")

    def _drop_packet(self, pkt):
        print(f"🛡️ Dropping suspicious packet from {pkt[IP].src}")

async def main():
    agent = SecurityAgent()
    await agent.monitor_network()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
