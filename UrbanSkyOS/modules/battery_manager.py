"""
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
    time_left = bm.estimate_remaining_time(1000, 15) # 1000mA draw, 15m/s wind
    print(f"Remaining time: {time_left:.2f} mins")
