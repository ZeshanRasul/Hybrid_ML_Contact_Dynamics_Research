import torch
import numpy as np

from hybrid_ml_contact_dynamics.ml.models.model_restitution import RestitutionPredictor
from hybrid_ml_contact_dynamics.ml.data.build_restitution_dataset import Restitution_Data_Builder

def main():
    model = RestitutionPredictor()
    data_builder = Restitution_Data_Builder()
    data_builder.save()
    data = data_builder.load()

    X = np.zeros((6717, 2))
    y = list()
        
    X[:,0] = data['v_prev']
    X[:,1] = data['v_post']
    y = data['e_actual']

    e_analytic = data['v_post'] / (-data['v_prev'])
    analytic_mse = np.mean((e_analytic - y)**2)

    print(f"Analytic error is: {analytic_mse}")

    X = np.asarray(X, dtype=np.float32)
    X = torch.from_numpy(X)
    y = np.asarray(y, dtype=np.float32)
    y = torch.from_numpy(y)

    loss_fn = torch.nn.MSELoss()

    print(y.min())
    print(y.max())
    print(y.mean())
    print(y.std())
    

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    #model.train()

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
