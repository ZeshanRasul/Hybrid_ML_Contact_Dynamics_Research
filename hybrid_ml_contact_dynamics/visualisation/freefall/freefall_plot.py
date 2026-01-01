import matplotlib.pyplot as plt
import numpy as np
from hybrid_ml_contact_dynamics.experiments.freefall.trajectory_buffer import Trajectory_Buffer

class freefall_plot:
    traj: Trajectory_Buffer
    data: list

    def __init__(self):
        self.traj = Trajectory_Buffer()

    def plot(self):
        self.data = self.traj.load()
        plt.plot(self.data['time'], self.data['position'])
        plt.show()



def main():
    plotter = freefall_plot()
    plotter.plot()