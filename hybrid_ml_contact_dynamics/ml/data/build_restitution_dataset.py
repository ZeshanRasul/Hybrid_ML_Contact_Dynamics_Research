import json
import numpy as np
from pathlib import Path

class Restitution_Data_Builder:
    prev_v: list
    post_v: list

    def __init__(self):
        self.training_data = list()
        self.y_windows = list()
        self.y_window = list()
        self.h_windows = list()
        self.h_window = list()
        self.e_truth = list()
        self.e_obs = list()
        self.e_analytic = list()
        self.e_true_obs = list()
        date = "02-01-2026,21:24:18"
        y_count = 0
        h_count = 0
        e_count = 0
        e_analytic_count = 0
        e_obs_count = 0
        e_true_obs_count = 0
        for i in range(300):
            with open(f"hybrid_ml_contact_dynamics/experiments/freefall/runs/{i}/{date}/results/validation.json") as f:
                loaded_json = json.load(f)
                y_windows = loaded_json['Y Windows']
                h_windows = loaded_json['H Windows']
                e = loaded_json['e true']
                e_analytic = loaded_json['e analytic']
                e_obs = loaded_json['e obs']
                e_true_obs = loaded_json['e true obs']


                y_windows = np.asarray(y_windows)
                for j in range(0, len(y_windows) - 1):
                    y_window = list()
                    for k in range(0, len(y_windows[j]) - 1):
                        y_window.append(y_windows[j][k])
                    self.y_windows.append(y_windows[j])
                    y_count += 1


                for j in range(len(h_windows) - 1):
                    for k in range(len(h_windows[0]) - 1):
                        self.h_window.append(h_windows[j])
                    self.h_windows.append(h_windows[j])
                    h_count += 1

                for h in range(len(e) - 1):
                    self.e_truth.append(e[h])
                    e_count += 1

                for k in range(len(e_analytic) - 1):
                    self.e_analytic.append(e_analytic[k])
                    e_analytic_count += 1
                
                for n in range(len(e_obs) - 1):
                    self.e_obs.append(e_obs[n])
                    self.e_true_obs.append(e_true_obs[n])
                    e_obs_count += 1
                    e_true_obs_count += 1

        print(f"y_count = {y_count}")

        print(f"h_count = {h_count}")

        print(f"e count = {e_count}")

        print(f"e analytic count = {e_analytic_count}")

        print(f"e obs count = {e_obs_count}")
        print(f"e true obs count = {e_true_obs_count}")


        self.y_windows = np.asarray(self.y_windows, dtype=object)
        self.h_windows = np.asarray(self.h_windows, dtype=object)

    def save(self):
        Path(f"hybrid_ml_contact_dynamics/ml/data/restitution_data").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f"hybrid_ml_contact_dynamics/ml/data/restitution_data/training_data.npz", y_windows=self.y_windows, h_windows=self.h_windows, e_true=self.e_truth, e_analytic=self.e_analytic, e_obs=self.e_obs, e_true_obs=self.e_true_obs)
        

    def load(self):
        return np.load(f"hybrid_ml_contact_dynamics/ml/data/restitution_data/training_data.npz", allow_pickle=True)

def main():
    data_builder = Restitution_Data_Builder()
   # data_builder.save()