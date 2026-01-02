import torch
import numpy as np

from hybrid_ml_contact_dynamics.ml.models.model_restitution import RestitutionPredictor
from hybrid_ml_contact_dynamics.ml.data.build_restitution_dataset import Restitution_Data_Builder

def main():
    model = RestitutionPredictor()
    data_builder = Restitution_Data_Builder()
    data_builder.save()
    data = data_builder.load()
    X = list()
    for i in range(len(data['v_prev'])):
        X.append(data['v_prev'][i])
        X.append(data['v_post'][i])
    X = np.asarray(X, dtype=np.float32)
    X = torch.from_numpy(X)
    y = data['e_actual']
    print(len(data['e_actual']))
    y = np.asarray(y, dtype=np.float32)
    y = torch.from_numpy(y)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()

    running_loss = 0
    epochs = 100
    for i in range(epochs):
        optimizer.zero_grad()

        predictions = model(X)
        
        loss = loss_fn(predictions, y)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if i % 10 == 0:
            last_loss = running_loss / 50
            print(last_loss)
            running_loss = 0
