import torch
import numpy as np

from hybrid_ml_contact_dynamics.ml.models.model_restitution import RestitutionPredictor
from hybrid_ml_contact_dynamics.ml.data.build_restitution_dataset import Restitution_Data_Builder

def main():
    model = RestitutionPredictor()
    data_builder = Restitution_Data_Builder()
    data_builder.save()
    data = data_builder.load()

    X1 = np.zeros((6040))
    X2 = np.zeros((6040))
        
    y_windows = list()
    h_windows = list()
    y_windows = data['y_windows']
    h_windows = data['h_windows']
    y_windows = np.asarray(y_windows, dtype=np.float64)
    h_windows = np.asarray(h_windows, dtype=np.float64)

    X1 = y_windows
    X2 = h_windows

    X1 = np.asarray(X1, dtype=np.float32)
    X1 = torch.from_numpy(X1)
    X2 = np.asarray(X2, dtype=np.float32)
    X2 = torch.from_numpy(X2)

    loss_fn = torch.nn.MSELoss()

    # print(X1.min())
    # print(X1.max())
    # print(X1.mean())
    # print(X1.std())

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()

    running_loss = 0
    epochs = 100
    for i in range(epochs):
        optimizer.zero_grad()

        predictions = model(X1)
        
      #  loss = loss_fn(predictions, y)
     #   loss.backward()

        optimizer.step()

    #    running_loss += loss.item()

        if i % 10 == 0:
            last_loss = running_loss / 50
            print(last_loss)
            running_loss = 0
