from hybrid_ml_contact_dynamics.physics.types import Vec2
from hybrid_ml_contact_dynamics.physics.rigidbody import RigidBody2D

class Circle:
    position: Vec2
    radius: float
    rigidbody: RigidBody2D
    restitution: float

    def __init__(self, position: Vec2, radius: float):
        self.position = position
        self.radius = radius
        self.rigidbody = RigidBody2D(self.position, 1.0)
        self.restitution = 0.6

    def set_position(self, position):
        self.rigidbody.set_position(position)

    def set_mass(self, mass):
        self.rigidbody.set_mass(mass)

    def set_velocity(self, velocity):
       self.rigidbody.set_velocity(velocity)

    def set_restitution(self, e):
        self.restitution = e

    def add_force(self, force):
        self.rigidbody.add_force(force)

    def get_rigidbody(self):
        return self.rigidbody

    def get_position(self):
        return self.rigidbody.position

    def get_velocity(self):
        return self.rigidbody.velocity
    
    def get_acceleration(self):
        return self.rigidbody.acceleration
    
    def get_restitution(self):
        return self.restitution;

    def get_mass(self):
        return self.rigidbody.mass
    
    def get_radius(self):
        return self.radius