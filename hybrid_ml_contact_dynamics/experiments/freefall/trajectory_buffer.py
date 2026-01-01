import numpy as np
from pathlib import Path
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
    
    def save(self, date):
        Path(f"hybrid_ml_contact_dynamics/experiments/freefall/runs/{date}/results/").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f"hybrid_ml_contact_dynamics/experiments/freefall/runs/{date}/results/traj.npz", time=self.time, position=self.position, velocity=self.velocity)

    def load(self, date):
        return np.load(f"hybrid_ml_contact_dynamics/experiments/freefall/runs/{date}/results/traj.npz")