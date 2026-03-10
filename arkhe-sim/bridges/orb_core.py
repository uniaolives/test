import json
import hashlib
import time

class OrbPayload:
    def __init__(self, lambda_2, phi_q, h_value, origin_time, target_time):
        self.lambda_2 = lambda_2
        self.phi_q = phi_q
        self.h_value = h_value
        self.origin_time = origin_time
        self.target_time = target_time
        self.orb_id = hashlib.sha256(str(time.time()).encode()).digest()

    @classmethod
    def create(cls, lambda_2, phi_q, h_value, origin_time, target_time):
        return cls(lambda_2, phi_q, h_value, origin_time, target_time)

    def to_json(self):
        return json.dumps({
            'lambda_2': self.lambda_2,
            'phi_q': self.phi_q,
            'h_value': self.h_value,
            'origin_time': self.origin_time,
            'target_time': self.target_time,
            'orb_id': self.orb_id.hex()
        })

    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        return cls(data['lambda_2'], data['phi_q'], data['h_value'], data['origin_time'], data['target_time'])

    def to_bytes(self):
        return self.to_json().encode()

    @classmethod
    def from_bytes(cls, data):
        return cls.from_json(data.decode())

    def informational_mass(self):
        return self.lambda_2 * self.phi_q

    def is_retrocausal(self):
        return self.target_time < self.origin_time

    def temporal_span(self):
        return abs(self.target_time - self.origin_time)
