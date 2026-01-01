import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from hybrid_ml_contact_dynamics.experiments.freefall.trajectory_buffer import Trajectory_Buffer

class freefall_plot:
    traj: Trajectory_Buffer
    data: list
    validation_data: list

    def __init__(self):
        self.traj = Trajectory_Buffer()

    def plot(self, runarg, i):
        self.data = self.traj.load(i, f"{runarg}")
        self.validation_data = list()
        with open(f"hybrid_ml_contact_dynamics/experiments/freefall/runs/{i}/{runarg}/results/validation.json") as f:
            self.validation_data = json.load(f)

        Path(f"hybrid_ml_contact_dynamics/experiments/freefall/runs/{i}/{runarg}/plots/").mkdir(parents=True, exist_ok=True)
        plt.suptitle("Circle Position versus Time")
        plt.plot(self.data['time'], self.data['position'])
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        plt.savefig(f"hybrid_ml_contact_dynamics/experiments/freefall/runs/{i}/{runarg}/plots/timevsposition")
        plt.close()
        
        plt.suptitle("Circle Velocity versus Time")
        plt.plot(self.data['time'], self.data['velocity'])
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (m/s)')
        plt.savefig(f"hybrid_ml_contact_dynamics/experiments/freefall/runs/{i}/{runarg}/plots/timevsvelocity")
        plt.close()
        
        plt.suptitle("Circle Position versus Velocity")
        plt.xlabel('Position (m)')
        plt.ylabel('Velocity (m/s)')
        plt.plot(self.data['position'], self.data['velocity'])
        plt.savefig(f"hybrid_ml_contact_dynamics/experiments/freefall/runs/{i}/{runarg}/plots/positionvsvelocity")
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", "--opts")
    parser.add_argument("--run_count")

    args = parser.parse_args()
    rundir = args.run
    count = int(args.run_count)
    plotter = freefall_plot()
    for i in range(count):
        plotter.plot(rundir, i)