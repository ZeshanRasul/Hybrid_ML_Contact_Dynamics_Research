from hybrid_ml_contact_dynamics.physics.rigidbody import RigidBody2D

class Simulation:   
    bodies: list[RigidBody2D]

    def __init__(self):
        self.bodies = list()

    def add_rigidbody(self, body: RigidBody2D):
        self.bodies.append(body)

    def step(self, delta_time: float):
        for body in self.bodies:
            body.integrate(delta_time)
