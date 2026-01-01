from physics.types import Vec2
from physics.rigidbody import RigidBody2D

class Circle:
    position: Vec2
    radius: float
    rigidbody: RigidBody2D

    def __init__(self, position: Vec2, radius: float):
        self.position = position
        self.radius = radius
        self.rigidbody = RigidBody2D(self.position, 1.0)

    def set_position(self, position):
        self.rigidbody.set_position(position)

    def set_mass(self, mass):
        self.rigidbody.set_mass(mass)

    def set_velocity(self, velocity):
       self.rigidbody.set_velocity(velocity)

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
    
    def get_mass(self):
        return self.rigidbody.mass