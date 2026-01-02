import torch
import numpy as np

from hybrid_ml_contact_dynamics.ml.models.model_restitution import RestitutionPredictor
from hybrid_ml_contact_dynamics.ml.data.build_restitution_dataset import Restitution_Data_Builder

def main():
    model = RestitutionPredictor()
    data_builder = Restitution_Data_Builder()
    data_builder.save()
    data = data_builder.load()


    y_windows = list()
    h_windows = list()
    y_windows = np.asarray(data['y_windows'], dtype=object)
    h_windows = np.asarray(data['h_windows'], dtype=object)

    print(y_windows.shape)
    print(h_windows.shape)
    y = data['e_true']
    
    e_analytic = np.asarray(data['e_analytic'])
    e_true_valid = y[:len(e_analytic)]

    mse_analytic = np.mean((e_analytic - e_true_valid)**2)

    e_obs = np.asarray(data['e_obs'])
    e_true_obs = np.asarray(data['e_true_obs'])
    mse_obs = np.mean((e_obs - e_true_obs)**2)



    print(f"mse_analytic is = {mse_analytic}")
    print(f"mse observed is = {mse_obs}")
    print(len(y))
    X1 = np.copy(y_windows)
    X2 = h_windows

    print(len(X1))
    X1 = np.asarray(X1, dtype=np.float32)
    X2 = np.asarray(X2, dtype=np.float32)

    X = np.zeros((10, 1773))
    X = np.concatenate([X1, X2], axis = 1)
    X = torch.from_numpy(X)
    loss_fn = torch.nn.MSELoss()
    
    y = np.asarray(y, dtype=np.float32)
    y_mean = np.mean(y)
    mse_mean = np.mean((y - y_mean)**2)

    print(f"mse mean is: {mse_mean}")
    y = y.reshape(1773, 1)
    y = torch.from_numpy(y)
    print(X.shape)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()

    running_loss = 0
    epochs = 1000
    for i in range(epochs):
        optimizer.zero_grad()

        predictions = model(X)
        # print(predictions.min())
        # print(predictions.max())
    #    print(predictions.mean())
        # print(predictions.std())


        loss = loss_fn(predictions, y)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if i % 10 == 0:
            last_loss = running_loss / 10
            print(last_loss)
            running_loss = 0
