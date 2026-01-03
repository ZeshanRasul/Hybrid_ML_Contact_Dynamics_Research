import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from hybrid_ml_contact_dynamics.experiments.freefall.trajectory_buffer import Trajectory_Buffer

def ecdf(x):
    x = np.asarray(x)
    x = np.sort(x)
    n = x.size
    y = np.arange(1, n + 1)  / n
    return x, y

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

        plt.plot([low, clip_max], [low, clip_max], linestyle="--", linewidth=1)
        plt.xscale('log')
        plt.yscale('log')

        improvement_rate = improved.mean() * 100
        plt.legend(bbox_to_anchor=(1.02, 0), loc="center left", frameon=False, title="Sample Outcome")
        plt.title(f"Per-Sample Error Comparison (Improved {improvement_rate:.1f}%)", pad=8)

        plt.xlabel('Baseline Squared Error')
        plt.ylabel('Hybrid Squared Error')

        plt.tight_layout()
        plt.savefig("hybrid_ml_contact_dynamics/ml/training/freefall/03-01-2026,03:34:35/baselinevshybriderr")
        plt.close()

        delta = err_hybrid - err_obs

        lo, hi = np.quantile(delta, [0.01, 0.99])
        pad = 0.05 * (hi - lo)
        lo -= pad
        hi += pad

        plt.figure(figsize=(7, 4))
        neg = delta[delta < 0]
        pos = delta[delta > 0]

        plt.hist(neg, bins=50, alpha=0.8, label="Improved (Δ<0)")
        plt.hist(pos, bins=50, alpha=0.6, label="Regressed (Δ>0)")
        plt.legend(frameon=False)        
        plt.axvline(0.0, linestyle="--", linewidth=1, alpha=0.8, label="No change")
        plt.axvline(np.median(delta), linestyle=":", linewidth=1, alpha=0.8,
            label=f"Median Δ = {np.median(delta):.2e}")
        plt.xlabel("Δ squared error (hybrid − baseline)")
        plt.ylabel("Count")

        plt.xlim(lo, hi)
        plt.xscale("symlog", linthresh=1e-6)
        plt.title("Per-sample error change")
        plt.tight_layout()
        plt.savefig("hybrid_ml_contact_dynamics/ml/training/freefall/03-01-2026,03:34:35/deltasquarederror")
        plt.close()

        plt.figure(figsize=(6, 4))

        x_b, y_b = ecdf(err_obs)
        x_h, y_h = ecdf(err_hybrid)

        eps = 1e-12

        ps = np.linspace(0.0, 1.0, 300)

        qb = np.quantile(err_obs, ps)
        qh = np.quantile(err_hybrid, ps)

        qb = np.clip(qb, eps, None)
        qh = np.clip(qh, eps, None)


        x_b = np.clip(x_b, eps, None)
        x_h = np.clip(x_h, eps, None)

        plt.plot(qb, ps, label="Baseline", linewidth=1.5)
        plt.plot(qh, ps, label="Hybrid", linewidth=1.5)

        plt.xscale('log')
        plt.ylim(0.0, 1.0)
        plt.xlabel("Squared error")
        plt.ylabel("Cumulative probability")
        plt.title("Error ECDF (test set)")
        plt.legend(frameon=False)
        for p in [0.5, 0.9]:
            vb = max(np.quantile(err_obs, p), eps)
            vh = max(np.quantile(err_hybrid, p), eps)
            plt.axvline(vb, linestyle=":", linewidth=1, alpha=0.25)
            plt.axvline(vh, linestyle=":", linewidth=1, alpha=0.25)

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