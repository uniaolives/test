# modules/robotics/node/python/drone_node.py
import asyncio
from pymavlink import mavutil
# Ajustar import para o novo caminho do core
import sys
import os
# Add root to path for metalanguage
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
# Add core/python for core logic
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../core/python')))

from metalanguage.anl import Node, Handover

class DroneNode(Node):
    def __init__(self, connection_string):
        super().__init__(node_type="Drone", node_id="drone_1")
        self.conn = mavutil.mavlink_connection(connection_string)
        self.position = (0.0, 0.0, 0.0)
        self.battery = 100.0

        # Handover para receber comandos
        # Em ANL v0.7, handovers são registrados no sistema, mas aqui seguimos o exemplo
        # como uma funcionalidade do nó.

    def handle_goto(self, target):
        """Handover: recebe coordenadas e envia comando ao drone"""
        self.conn.mav.command_long_send(
            self.conn.target_system, self.conn.target_component,
            mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0, 0, 0, 0, 0,
            target[0], target[1], target[2])
        return {"status": "sent"}

    async def telemetry_loop(self):
        while True:
            # Simulação de recebimento de mensagens MAVLink
            msg = self.conn.recv_match(type=['GLOBAL_POSITION_INT', 'BATTERY_STATUS'], blocking=False)
            if msg:
                if msg.get_type() == 'GLOBAL_POSITION_INT':
                    self.position = (msg.lat/1e7, msg.lon/1e7, msg.alt/1000.0)
                elif msg.get_type() == 'BATTERY_STATUS':
                    self.battery = msg.battery_remaining
                # publica telemetria (exemplo simplificado)
                print(f"Telemetry: Pos={self.position}, Bat={self.battery}")
            await asyncio.sleep(0.1)

if __name__ == "__main__":
    # Exemplo de uso (requer simulador MAVLink rodando)
    # drone = DroneNode("tcp:localhost:5760")
    # asyncio.run(drone.telemetry_loop())
    print("DroneNode module ready.")
