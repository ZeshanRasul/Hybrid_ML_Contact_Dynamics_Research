import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from hybrid_ml_contact_dynamics.experiments.freefall.trajectory_buffer import Trajectory_Buffer

class freefall_plot:
    traj: Trajectory_Buffer
    data: list

    def __init__(self):
        self.traj = Trajectory_Buffer()

    def plot(self, runarg):
        self.data = self.traj.load(f"{runarg}")
        Path(f"hybrid_ml_contact_dynamics/experiments/freefall/runs/{runarg}/plots/").mkdir(parents=True, exist_ok=True)
        plt.suptitle("Circle Position versus Time")
        plt.plot(self.data['time'], self.data['position'])
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        plt.savefig(f"hybrid_ml_contact_dynamics/experiments/freefall/runs/{runarg}/plots/timevsposition")
        plt.close()
        
        plt.suptitle("Circle Velocity versus Time")
        plt.plot(self.data['time'], self.data['velocity'])
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (m/s)')
        plt.savefig(f"hybrid_ml_contact_dynamics/experiments/freefall/runs/{runarg}/plots/timevsvelocity")
        plt.close()
        
        plt.suptitle("Circle Position versus Velocity")
        plt.xlabel('Position (m)')
        plt.ylabel('Velocity (m/s)')
        plt.plot(self.data['position'], self.data['velocity'])
        plt.savefig(f"hybrid_ml_contact_dynamics/experiments/freefall/runs/{runarg}/plots/positionvsvelocity")
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", "--opts",)

    args = parser.parse_args()
    rundir = args.run
    plotter = freefall_plot()
    plotter.plot(rundir)