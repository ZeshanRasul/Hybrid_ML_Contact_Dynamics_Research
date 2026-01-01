import json
import numpy as np
from pathlib import Path

class Restitution_Data_Builder:
    prev_v: list
    post_v: list

    def __init__(self):
        self.training_data = list()
        self.true_y = list()
        self.prev_v = list()
        self.post_v = list()
        date = "01-01-2026,22:30:13"
        for i in range(50):
            with open(f"hybrid_ml_contact_dynamics/experiments/freefall/runs/{i}/{date}/results/validation.json") as f:
                loaded_json = json.load(f)
                self.true_y.append(loaded_json['e actual mean'])
                self.prev_v.append(loaded_json['x input'])
                self.post_v.append(loaded_json['x input'])
                self.training_data.append(self.prev_v)
                self.training_data.append(self.post_v)


        self.training_data = np.asarray(self.training_data, dtype=object)
        self.true_y = np.asarray(self.true_y, dtype=np.float64)

    def save(self):
        Path(f"hybrid_ml_contact_dynamics/ml/data/restitution_data").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f"hybrid_ml_contact_dynamics/ml/data/restitution_data/training_data.npz", v_prev=self.training_data[0], v_post=self.training_data[0][:], e_actual=self.true_y)
        

    def load(self):
        return np.load(f"hybrid_ml_contact_dynamics/ml/data/restitution_data/training_data.npz")

def main():
    data_builder = Restitution_Data_Builder()
    data_builder.save()