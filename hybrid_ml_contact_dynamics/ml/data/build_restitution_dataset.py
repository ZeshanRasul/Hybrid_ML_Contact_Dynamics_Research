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
        date = "02-01-2026,14:13:21"
        y_count = 0
        h_count = 0
        for i in range(1000):
            with open(f"hybrid_ml_contact_dynamics/experiments/freefall/runs/{i}/{date}/results/validation.json") as f:
                loaded_json = json.load(f)
                y_windows = loaded_json['Y Windows']
                h_windows = loaded_json['H Windows']
                y_windows = np.asarray(y_windows)
                for i in range(0, len(y_windows) - 1):
                    for j in range(0, len(y_windows[0]) - 1):
                        print(len(y_windows[0]))
                        self.y_window.append(y_windows[i][j])
                    self.y_windows.append(self.y_window)
                    self.y_window.clear()
                    y_count += 1


                for i in range(len(h_windows) - 1):
                    for j in range(len(h_windows[0]) - 1):
                        self.h_window.append(h_windows[i][j])
                    self.h_windows.append(self.h_window)
                    self.h_window.clear()
                    h_count += 1

                
        print(f"y_count = {y_count}")

        print(f"h_count = {h_count}")

        self.y_windows = np.asarray(self.y_windows, dtype=np.float64)
        self.h_windows = np.asarray(self.h_windows, dtype=np.float64)

    def save(self):
        Path(f"hybrid_ml_contact_dynamics/ml/data/restitution_data").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f"hybrid_ml_contact_dynamics/ml/data/restitution_data/training_data.npz", y_windows=self.y_windows, h_windows=self.h_windows)
        

    def load(self):
        return np.load(f"hybrid_ml_contact_dynamics/ml/data/restitution_data/training_data.npz")

def main():
    data_builder = Restitution_Data_Builder()
   # data_builder.save()