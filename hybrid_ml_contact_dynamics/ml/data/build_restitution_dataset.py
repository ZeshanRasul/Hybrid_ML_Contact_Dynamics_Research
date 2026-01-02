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
        date = "02-01-2026,01:19:25"
        count = 0
        for i in range(1000):
            with open(f"hybrid_ml_contact_dynamics/experiments/freefall/runs/{i}/{date}/results/validation.json") as f:
                loaded_json = json.load(f)
                x_input = loaded_json['x input']

                for i in range(len(x_input[:])):
                    self.prev_v.append(loaded_json['x input'][:][0][1])
                 #   print(loaded_json['x input'][:][0][1])
                    self.training_data.append(self.prev_v)
                    self.post_v.append(loaded_json['x input'][:][0][2])
                    self.training_data.append(self.post_v)
                    self.true_y.append(loaded_json['e actual mean']) 
                    count += 1
                #    print(loaded_json['x input'][:][0][2])

        print(f"count = {count}")

        self.training_data = np.asarray(self.training_data, dtype=np.float64)
        self.true_y = np.asarray(self.true_y, dtype=np.float64)

    def save(self):
        Path(f"hybrid_ml_contact_dynamics/ml/data/restitution_data").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f"hybrid_ml_contact_dynamics/ml/data/restitution_data/training_data.npz", v_prev=self.prev_v, v_post=self.post_v, e_actual=self.true_y)
        

    def load(self):
        return np.load(f"hybrid_ml_contact_dynamics/ml/data/restitution_data/training_data.npz")

def main():
    data_builder = Restitution_Data_Builder()
   # data_builder.save()