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
    y = data['e_true_obs']
    
    y_true = data['e_true']

    e_analytic = np.asarray(data['e_analytic'])
    e_true_valid = y_true[:len(e_analytic)]

    mse_analytic = np.mean((e_analytic - e_true_valid)**2)

    e_obs = data['e_obs']
    e_true_obs = data['e_true_obs']
  #  mse_obs = np.mean((e_obs - e_true_obs)**2)
    print(f"mse_analytic is = {mse_analytic}")
  #  print(f"mse observed is = {mse_obs}")
    print(len(y))
    X1 = np.copy(y_windows)
    X2 = h_windows

    r_true = e_true_obs - e_obs
    r_true = np.asarray(r_true, dtype=np.float32)
    r_true = r_true.reshape(2136, 1)
    r_true = torch.from_numpy(r_true)
    print(len(X1))
    X1 = np.asarray(X1, dtype=np.float32)
    X2 = np.asarray(X2, dtype=np.float32)

    X = np.zeros((10, 2136))
    X = np.concatenate([X1, X2], axis = 1)
    X = torch.from_numpy(X)
    loss_fn = torch.nn.MSELoss()
    
    y = np.asarray(y, dtype=np.float32)
    y_mean = np.mean(y)
    mse_mean = np.mean((y - y_mean)**2)

    print(f"mse mean is: {mse_mean}")
    y = y.reshape(2136, 1)
    y = torch.from_numpy(y)
    print(X.shape)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()

    running_loss = 0
    epochs = 10000
    for i in range(epochs):
        optimizer.zero_grad()

        r_hat = model(X)
        # print(predictions.min())
        # print(predictions.max())
    #    print(predictions.mean())
        # print(predictions.std())


        loss = loss_fn(r_hat, r_true)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if i % 50 == 0:
            last_loss = running_loss / 50
            print(last_loss)
            running_loss = 0

    e_obs = torch.from_numpy(np.asarray(e_obs, dtype=np.float32).reshape(-1, 1))
    e_hat = torch.clamp(e_obs + r_hat, 0, 1)
 #   e_hat = np.asarray(e_hat)
 #   e_obs = np.asarray(data['e_obs'])
    e_true_obs = np.asarray(data['e_true_obs'])
    e_true_obs = torch.from_numpy(np.asarray(e_true_obs, dtype=np.float32).reshape(-1, 1))

    mse_obs = torch.mean((e_obs - e_true_obs)**2)
    print(f"mse observed pre eval is = {mse_obs}")
    mse_residual = torch.mean((e_hat - e_true_obs)**2)
    print(f"mse residual pre eval = {mse_residual}")

    model.eval()
    with torch.no_grad():
        r_hat = model(X)
        e_hat = torch.clamp(e_obs + r_hat, 0, 1)
    
    clamp_rate = 0
    with torch.no_grad():
        e_raw = e_obs + r_hat
        clamp_rate = ((e_raw < 0.0) | (e_raw > 1.0)).float().mean().item()

    print(f"clamp rate = {clamp_rate}")

    mse_obs = torch.mean((e_obs - e_true_obs)**2)
    print(f"mse observed post eval is = {mse_obs}")
    mse_residual = torch.mean((e_hat - e_true_obs)**2)
    print(f"mse residual post eval = {mse_residual}")

    r_true_np = (e_true_obs - e_obs).cpu().numpy()
    r_hat_np = r_hat.cpu().numpy()
    print(r_true_np.mean(), r_true_np.std())
    print(r_hat_np.mean(), r_hat_np.std())