import numpy as np
from hybrid_ml_contact_dynamics.physics.types import vec2

class Trajectory_Buffer():
    def __init__(self):
        self.time = list()
        self.position = list()
        self.velocity = list()
        self.buffer = list()

    def record(self, time: float, position, velocity):
        self.time.append(time)
        self.position.append(position)
        self.velocity.append(velocity)
        self.buffer.append(vec2(time, 0))
        self.buffer.append(position)
        self.buffer.append(velocity)

    def finalise(self):
        return np.stack(self.buffer)
    
    def save(self):
        np.savez_compressed("hybrid_ml_contact_dynamics/experiments/freefall/results/traj.npz", time=self.time, position=self.position, velocity=self.velocity)

    def load(self):
        return np.load("hybrid_ml_contact_dynamics/experiments/freefall/results/traj.npz")