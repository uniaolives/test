import math

def haversine(lat1, lon1, lat2, lon2):
    """Calculates the distance in km between two points on Earth."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def delay_to_dist(rtt_ms):
    """
    Converts RTT in ms to distance in km.
    Roughly 100km per 1ms RTT (speed of light in fiber is ~2/3 c).
    """
    return (rtt_ms / 2.0) * 100.0
