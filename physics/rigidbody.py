from physics.types import Vec2, vec2

class RigidBody2D:
    position: Vec2
    mass: float
    inv_mass: float
    velocity: Vec2
    acceleration: Vec2
    force_accum: Vec2

    def __init__(self, position: Vec2, mass: float):
        self.position = position
        self.set_mass(mass)
        self.velocity = vec2(0.0, 0.0)
        self.acceleration = vec2(0.0, 0.0)
        self.force_accum = vec2(0.0, 0.0)

    def set_position(self, position: Vec2):
        self.position = position

    def set_mass(self, mass: float):
        if (mass == float('inf')):
            self.mass = mass
            self.inv_mass = 0.0
            return
        elif (mass <= 0.0):
            raise ValueError("Mass must be greater than 0.0")     
        self.mass = mass
        self.inv_mass = 1.0 / mass

    def set_velocity(self, velocity: Vec2):
        self.velocity = velocity

    def add_force(self, force: Vec2):
        self.force_accum = self.force_accum + force

    def integrate(self, delta_time: float):
        self.acceleration = self.force_accum * self.inv_mass
        self.velocity = self.velocity + self.acceleration * delta_time
        self.position = self.position + self.velocity * delta_time
        self.force_accum = vec2(0.0, 0.0)