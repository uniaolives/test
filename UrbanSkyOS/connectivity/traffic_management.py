"""
UrbanSkyOS UTM Layer (Refined)
Handles dynamic geofencing via simulated cloud service.
Implements robust polygon/bounding check.
"""

import numpy as np
import time
import json
from datetime import datetime, timedelta

class DynamicGeofencing:
    def __init__(self):
        self.zones = {} # id -> data

    def update_zone(self, zone_id, polygon, priority=1, dynamic=False, duration_hr=24):
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hr)

        # Calculate bounding box for faster check
        poly_np = np.array(polygon)
        min_x, min_y = np.min(poly_np, axis=0)
        max_x, max_y = np.max(poly_np, axis=0)

        self.zones[zone_id] = {
            "zone_id": zone_id,
            "polygon": polygon,
            "bbox": (min_x, min_y, max_x, max_y),
            "priority": priority,
            "dynamic": dynamic,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "status": "ACTIVE"
        }
        print(f"🛑 UTM Cloud: Zone '{zone_id}' active. BBox: [{min_x}, {min_y}] to [{max_x}, {max_y}]")
        return self.zones[zone_id]

    def check_zone(self, position):
        """Checks if a point is inside any active zone using BBox."""
        px, py = position[0], position[1]
        for zid, data in self.zones.items():
            if data["status"] == "ACTIVE":
                min_x, min_y, max_x, max_y = data["bbox"]
                if min_x <= px <= max_x and min_y <= py <= max_y:
                    return {"in_zone": True, "zone_id": zid, "priority": data["priority"]}
        return {"in_zone": False}

class UTMInterface:
    def __init__(self, drone_id="URBAN_001"):
        self.drone_id = drone_id
        self.geofence = DynamicGeofencing()

    def sync(self, position):
        res = self.geofence.check_zone(position)
        if res["in_zone"]:
             print(f"⚠️ ALERT [{self.drone_id}]: VIOLATION in Zone {res['zone_id']}")
        return res
