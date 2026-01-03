import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from hybrid_ml_contact_dynamics.experiments.freefall.trajectory_buffer import Trajectory_Buffer

def plot_cdf(x, label):
    x = np.sort(x)
    y = np.linspace(0, 1, len(x))
    plt.plot(x, y, label=label)

plot_runs = False

class freefall_plot:
    traj: Trajectory_Buffer
    data: list
    validation_data: list
    model_results: list
    def __init__(self):
        self.traj = Trajectory_Buffer()

    def plot(self, runarg, i):
        self.data = self.traj.load(i, f"{runarg}")
        self.validation_data = list()
        self.model_results = list()
        with open(f"hybrid_ml_contact_dynamics/ml/training/freefall/03-01-2026,03:34:35/model_results.json") as f:
            self.model_results = json.load(f)
        
        if (plot_runs):
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

        e_obs = np.asarray(self.model_results['Observed Restitution'])
        e_true = np.asarray(self.model_results['Ground Truth Restitution'])
        e_hybrid = np.asarray(self.model_results['Predicted Restitution'])

        err_obs = (e_obs - e_true)**2
        err_hybrid = (e_hybrid - e_true)**2

        low = 0.0
        clip_max = np.percentile(np.concatenate([err_obs, err_hybrid]), 99)
        err_obs_v = np.clip(err_obs, 0.0, clip_max)
        err_hybrid_v = np.clip(err_hybrid, 0.0, clip_max)
        plt.figure(figsize=(6, 6))
        max_err = max(err_obs_v.max(), err_hybrid_v.max())
        improved_frac = (err_hybrid < err_obs).mean()
        
        improved = err_hybrid < err_obs
        regressed = ~improved

        plt.scatter(err_obs_v[improved], err_hybrid_v[improved], s=12, alpha=0.7, label='Improved')
        plt.scatter(err_obs_v[regressed], err_hybrid_v[regressed], s=12, alpha=0.4, label='Regressed')

        lims = [ min(err_obs.min(), err_hybrid.min()), max(err_obs.max(), err_hybrid.max()) ]

       # plt.text(0.05 * clip_max, 0.9 * clip_max, f"Improved samples: {improved_frac:.1%}")
        plt.plot([low, clip_max], [low, clip_max], linestyle="--", linewidth=1)
      #  plt.xlim(lims)
      #  plt.ylim(lims)
        plt.xscale('log')
        plt.yscale('log')

        improvement_rate = improved.mean() * 100
        plt.legend(bbox_to_anchor=(1.02, 0), loc="center left", frameon=False, title="Sample Outcome")
        plt.title(f"Per-Sample Error Comparison (Improved {improvement_rate:.1f}%)", pad=8)

        plt.xlabel('Baseline Squared Error')
        plt.ylabel('Hybrid Squared Error')

        # plt.text(
        #     0.02, 0.02, 
        #     f"Improved samples: {improvement_rate:.1f}%", 
        #     transform=plt.gca().transAxes,
        #     va="bottom",
        #     ha="right",
        #     fontsize=9
        #     )

        plt.tight_layout()
        plt.savefig("hybrid_ml_contact_dynamics/ml/training/freefall/03-01-2026,03:34:35/baselinevshybriderr")
        plt.close()


        delta = err_hybrid - err_obs

        plt.figure(figsize=(8, 4))
        plt.hist(delta, bins=50)
        plt.axvline(0.0)
        plt.xlabel("Δ squared error (hybrid − baseline)")
        plt.ylabel("Count")
        plt.title("Per-sample error change")
        plt.tight_layout()
        plt.savefig("hybrid_ml_contact_dynamics/ml/training/freefall/03-01-2026,03:34:35/deltasquarederror")
        plt.close()

 

        plt.figure(figsize=(7, 5))
        plot_cdf(err_obs, "Baseline")
        plot_cdf(err_hybrid, "Hybrid")

        plt.xlabel("Squared error")
        plt.ylabel("Cumulative probability")
        plt.title("Error CDF (test set)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("hybrid_ml_contact_dynamics/ml/training/freefall/03-01-2026,03:34:35/cdf")
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", "--opts")
    parser.add_argument("--run_count")

    args = parser.parse_args()
    rundir = "02-01-2026,22:13:42"
    count = 1300
    plotter = freefall_plot()
    if (not plot_runs):
        count = 1
    for i in range(count):
        plotter.plot(rundir, i)