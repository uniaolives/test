"""
UrbanSkyOS Lidar Service (Refined)
Simulates a rotating LiDAR sensor with obstacle detection (ray casting),
intensity falloff, and noise.
Provides data for PointCloudViewer, SensorTelemetryPanel, and SignalScope.
"""

import numpy as np
import random
import time

class LidarService:
    def __init__(self, scan_rate=10.0, max_range=100.0, angular_resolution=0.5):
        self.rate = scan_rate
        self.max_range = max_range
        self.angle_step = np.radians(angular_resolution)
        self.noise_std = 0.05
        self.signal_base = 100.0

        # Simulated obstacles: Buildings (boxes) and Trees (cylinders)
        self.obstacles = self._generate_obstacles()

        self.last_points = []
        self.last_intensities = []

    def _generate_obstacles(self):
        """Generates static obstacles for the simulation."""
        obstacles = []
        # Buildings (type, x, y, width, depth, height)
        for i in range(5):
            obstacles.append({
                'type': 'building',
                'x': random.uniform(-50, 50),
                'y': random.uniform(-50, 50),
                'width': random.uniform(5, 20),
                'depth': random.uniform(5, 20),
                'height': random.uniform(10, 50)
            })
        # Trees (type, x, y, radius)
        for i in range(10):
            obstacles.append({
                'type': 'tree',
                'x': random.uniform(-60, 60),
                'y': random.uniform(-60, 60),
                'radius': random.uniform(1, 3)
            })
        return obstacles

    def generate_scan(self, drone_pos):
        """
        Executes a 360 scan from the drone position.
        Simulates ray casting against obstacles.
        """
        angles = np.arange(0, 2*np.pi, self.angle_step)
        self.last_points = []
        self.last_intensities = []

        drone_x, drone_y = drone_pos[0], drone_pos[1]

        for theta in angles:
            min_dist = self.max_range
            hit_type = None

            for obs in self.obstacles:
                dist = self.max_range + 1
                if obs['type'] == 'building':
                    # Simplified 2D intersection with AABB
                    # Ray: (dx, dy) = (cos(theta), sin(theta))
                    # Check if ray enters building bounds
                    dx = np.cos(theta)
                    dy = np.sin(theta)

                    # Proximity check
                    dist_to_center = np.sqrt((obs['x']-drone_x)**2 + (obs['y']-drone_y)**2)
                    if dist_to_center < 30: # Only check nearby
                         # Simple distance if angle matches roughly
                         angle_to_obs = np.arctan2(obs['y']-drone_y, obs['x']-drone_x)
                         if abs(angle_to_obs - theta) < 0.2:
                              dist = dist_to_center - obs['width']/2

                elif obs['type'] == 'tree':
                    # Simplified cylinder intersection
                    dx = np.cos(theta)
                    dy = np.sin(theta)
                    ox, oy = obs['x'] - drone_x, obs['y'] - drone_y

                    # Distance of ray to circle center
                    proj = ox * dx + oy * dy
                    if proj > 0:
                        perp_dist_sq = (ox**2 + oy**2) - proj**2
                        if perp_dist_sq < obs['radius']**2:
                            dist = proj - np.sqrt(obs['radius']**2 - perp_dist_sq)

                if dist < min_dist:
                    min_dist = dist
                    hit_type = obs['type']

            # Apply range noise
            measured_dist = min_dist + np.random.normal(0, self.noise_std)
            if measured_dist < self.max_range:
                pt = [
                    drone_x + measured_dist * np.cos(theta),
                    drone_y + measured_dist * np.sin(theta),
                    drone_pos[2] # current altitude
                ]
                self.last_points.append(pt)

                # Signal strength (intensity): decays with distance
                intensity = self.signal_base * np.exp(-0.02 * measured_dist)
                if hit_type == 'building': intensity *= 0.9
                elif hit_type == 'tree': intensity *= 0.6
                else: intensity *= 0.3 # Noise/Atmosphere

                self.last_intensities.append(intensity)

        return np.array(self.last_points)

    def get_point_cloud_data(self):
        """Data for PointCloudViewer."""
        return {"points": self.last_points, "intensities": self.last_intensities}

    def get_telemetry(self):
        """Data for SensorTelemetryPanel."""
        return {
            "status": "OPERATIONAL" if len(self.last_points) > 0 else "SCANNING",
            "points_count": len(self.last_points),
            "max_range": self.max_range,
            "rate": self.rate
        }

    def get_signal_scope_data(self):
        """Data for SignalScope (intensities array)."""
        return self.last_intensities

if __name__ == "__main__":
    ls = LidarService()
    ls.generate_scan((0,0,10))
    print(ls.get_telemetry())
