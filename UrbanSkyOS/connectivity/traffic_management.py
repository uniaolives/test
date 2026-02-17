"""
UrbanSkyOS Connectivity & Cloud Layer
Handles communication with UTM (Unmanned Traffic Management).
Implements /update_zone endpoint for dynamic no-fly zones.
"""

import numpy as np
import time
import json
from datetime import datetime

class DynamicGeofencing:
    def __init__(self):
        self.no_fly_zones = [] # List of dicts

    def update_zone(self, zone_id, coordinates, start_time, end_time, reason="temporary"):
        """
        Simulates the /update_zone API logic.
        coordinates: list of [lon, lat] pairs (simplified as [x, y])
        """
        if len(coordinates) < 3:
            return {"error": "Invalid polygon"}

        new_zone = {
            "id": zone_id,
            "polygon": coordinates,
            "start": start_time,
            "end": end_time,
            "reason": reason,
            "status": "ACTIVE"
        }
        self.no_fly_zones.append(new_zone)
        print(f"🛑 UTM [Cloud]: Zone '{zone_id}' created ({reason}). Valid until {end_time}.")
        return {"status": "zone added", "zone_id": zone_id}

    def check_zone(self, pos, current_time=None):
        """
        Checks if position is inside any active zone.
        """
        if current_time is None: current_time = datetime.now().isoformat()

        # In a real system, would use shapely.geometry.Polygon
        # Simplified: check distance to first coordinate for simulation
        for zone in self.no_fly_zones:
             if zone["status"] == "ACTIVE":
                  # Simple radius check simulation
                  center = zone["polygon"][0]
                  dist = np.sqrt((pos[0]-center[0])**2 + (pos[1]-center[1])**2)
                  if dist < 10.0:
                       return {"in_zone": True, "zone_id": zone["id"], "reason": zone["reason"]}
        return {"in_zone": False}

class UTMInterface:
    def __init__(self, drone_id="URBAN_001"):
        self.drone_id = drone_id
        self.geofence = DynamicGeofencing()

    def sync(self, position):
        # Notify drone of its geofence status
        status = self.geofence.check_zone(position)
        if status["in_zone"]:
             print(f"⚠️ UTM Notify [{self.drone_id}]: Entering {status['zone_id']}! REASON: {status['reason']}")
        return status

    def detect_conflicts(self, own_trajectory, other_trajectories):
        # Implementation from previous steps
        return []

if __name__ == "__main__":
    utm = UTMInterface()
    res = utm.geofence.update_zone("Fire_01", [[30, 40], [35, 40], [35, 45]], "2026-02-18T10:00", "2026-02-18T12:00")
    print(res)
    print(utm.sync((31, 41)))
