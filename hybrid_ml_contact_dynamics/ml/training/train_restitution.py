import torch
import numpy as np

from hybrid_ml_contact_dynamics.ml.models.model_restitution import RestitutionPredictor
from hybrid_ml_contact_dynamics.ml.data.build_restitution_dataset import Restitution_Data_Builder

def main():
    model = RestitutionPredictor()
    data_builder = Restitution_Data_Builder()
    data_builder.save()
    data = data_builder.load()

    X1 = np.zeros((1903, 0))
 #   X2 = np.zeros((1903, 0))
 #   y = np.zeros((1903, 0))

    y_windows = list()
    h_windows = list()
    y_windows = data['y_windows']
    h_windows = data['h_windows']
    y_windows = np.asarray(y_windows, dtype=np.float64)
    h_windows = np.asarray(h_windows, dtype=np.float64)

    y = data['e_true']

    X1 = np.copy(y_windows)
    X2 = h_windows

    print(len(X1))
    X1 = np.asarray(X1, dtype=np.float32)
    X1 = torch.from_numpy(X1)
    X2 = np.asarray(X2, dtype=np.float32)
    X2 = torch.from_numpy(X2)

  #  X1 = X1.reshape(1903,)
    X = np.zeros((2, 1903))
    X = np.column_stack((X1, X2))
    X = torch.from_numpy(X)
    loss_fn = torch.nn.MSELoss()
    
    y = np.asarray(y, dtype=np.float32)
    y = y.reshape(1903, 1)
    y = torch.from_numpy(y)
   # print(len(X1))
    # print(X1.min())
    # print(X1.max())
    # print(X1.mean())
    # print(X1.std())

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()

    running_loss = 0
    epochs = 1000
    for i in range(epochs):
        optimizer.zero_grad()

        predictions = model(X)
        print(predictions.min())
        print(predictions.max())
        print(predictions.mean())
        print(predictions.std())


        loss = loss_fn(predictions, y)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if i % 10 == 0:
            last_loss = running_loss / 10
            print(last_loss)
            running_loss = 0
