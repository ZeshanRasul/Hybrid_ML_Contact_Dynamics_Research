import numpy as np
from pathlib import Path
from hybrid_ml_contact_dynamics.physics.types import vec2

class Trajectory_Buffer():
    def __init__(self):
        self.time = list()
        self.position = list()
        self.velocity = list()
        self.e = list()
        self.buffer = list()

    def record(self, time: float, position, velocity, e):
        self.time.append(time)
        self.position.append(position)
        self.velocity.append(velocity)
        self.e.append(e)
        self.buffer.append(vec2(time, 0))
        self.buffer.append(position)
        self.buffer.append(velocity)
        self.buffer.append(vec2(e, 0))

    def finalise(self):
        return np.stack(self.buffer)
    
    def save(self, i, date):
        Path(f"hybrid_ml_contact_dynamics/experiments/freefall/runs/{i}/{date}/results/").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f"hybrid_ml_contact_dynamics/experiments/freefall/runs/{i}/{date}/results/traj.npz", time=self.time, position=self.position, velocity=self.velocity, e=self.e)

    def load(self, i, date):
        return np.load(f"hybrid_ml_contact_dynamics/experiments/freefall/runs/{i}/{date}/results/traj.npz")