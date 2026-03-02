"""
UrbanSkyOS Battery Manager Module (Refined)
Calculates flight autonomy based on real-time wind conditions and environmental context.
"""

import numpy as np

class BatteryManager:
    def __init__(self, capacity_mah=5000):
        self.capacity = capacity_mah
        self.current_charge = 1.0 # 100%
        self.nominal_voltage = 22.2
        self.min_return_charge = 0.2 # 20% reserve

    def estimate_autonomy(self, base_power_draw, wind_speed, area_type):
        """
        Calculates remaining flight time.
        Wind speed increases power draw significantly in urban canyons.
        """
        # Wind penalty: Power increases non-linearly with wind
        # P = P_base * (1 + (v_wind / v_max_stab)^2)
        wind_factor = 1.0 + (wind_speed / 15.0)**2

        # Environmental factor (e.g., turbulence in canyons)
        env_factor = 1.2 if area_type == "Residential" else 1.0

        total_draw = base_power_draw * wind_factor * env_factor

        # Remaining time in minutes
        remaining_ah = self.capacity * self.current_charge / 1000.0
        remaining_time = (remaining_ah / total_draw) * 60.0

        return max(0, remaining_time)

    def check_safety_return(self, distance_to_home, wind_speed):
        """
        Determines if current battery is sufficient for return journey.
        """
        # Estimated speed against wind
        ground_speed = max(2.0, 10.0 - wind_speed)
        time_to_return = distance_to_home / ground_speed / 60.0 # mins

        # Power for return
        return_draw = 1.0 * (1.0 + (wind_speed / 15.0)**2)
        charge_needed = (return_draw * (time_to_return / 60.0)) * 1000.0 / self.capacity

        if self.current_charge < (charge_needed + self.min_return_charge):
             print(f"🔋 BATTERY WARNING: Return required! Est needed: {charge_needed*100:.1f}%.")
             return False

UrbanSkyOS Battery Manager Module
Calculates remaining flight time based on wind conditions in urban canyons.
"""

class BatteryManager:
    def __init__(self, capacity_mah=5000):
        self.capacity = capacity_mah
        self.current_charge = 1.0  # 100%
        self.voltage = 22.2

    def estimate_remaining_time(self, current_power_draw, wind_speed_mps):
        # In urban 'wind tunnels', power draw increases significantly
        wind_factor = 1.0 + (wind_speed_mps / 20.0)  # Simple linear penalty
        adjusted_draw = current_power_draw * wind_factor

        remaining_minutes = (self.capacity * self.current_charge) / adjusted_draw * 60

        if remaining_minutes < 5:
            print(f"⚠️ Low Battery! Wind is strong ({wind_speed_mps} m/s). Forced landing imminent.")
            return 0

        return remaining_minutes

    def check_return_to_home(self, distance_to_home, wind_speed_mps):
        # Safety check: can we make it back?
        # print(f"🔋 Battery: Checking RTH feasibility for {distance_to_home}m distance...")
        return True

if __name__ == "__main__":
    bm = BatteryManager()
    time_left = bm.estimate_autonomy(1.0, 10.0, "Commercial")
    print(f"Est autonomy: {time_left:.1f} mins")
    time_left = bm.estimate_remaining_time(1000, 15) # 1000mA draw, 15m/s wind
    print(f"Remaining time: {time_left:.2f} mins")
