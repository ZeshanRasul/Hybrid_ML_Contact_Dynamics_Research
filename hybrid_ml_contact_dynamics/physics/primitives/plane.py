import numpy as np
from hybrid_ml_contact_dynamics.physics.types import Vec2, vec2
from hybrid_ml_contact_dynamics.physics.rigidbody import RigidBody2D

class Plane:
    normal: Vec2
    offset: float
    rigidbody: RigidBody2D

    def __init__(self, normal: Vec2, offset: float):
        n = np.array(normal, dtype=float, copy=True)
        normalised_n = np.linalg.norm(n)
        if (normalised_n == 0):
            raise ValueError("Normal cannot be zero")
        self.normal = n / normalised_n
        self.offset = offset

    def get_normal(self):
        return self.normal
    
    def get_offset(self):
        return self.offset
