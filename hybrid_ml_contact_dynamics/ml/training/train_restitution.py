import torch
import numpy as np
import copy as copy
from sklearn.model_selection import train_test_split

from hybrid_ml_contact_dynamics.ml.models.model_restitution import RestitutionPredictor
from hybrid_ml_contact_dynamics.ml.data.build_restitution_dataset import Restitution_Data_Builder

def main():
    model = RestitutionPredictor()
    best_state = copy.deepcopy(model)
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

   # X_train, X_test, y_train, y_test = train_test_split(X, r_true, test_size=0.33, random_state=42)

    indices = np.random.permutation(2136)
    training_index, val_index, test_index = indices[:1500], indices[1500:1800], indices[1800:]
    X_train, X_val, X_test = X[training_index], X[val_index], X[test_index]
    y_train, y_val, y_test = r_true[training_index, :], r_true[val_index], r_true[test_index, :]

    e_obs = torch.from_numpy(np.asarray(e_obs, dtype=np.float32).reshape(-1, 1))
    e_obs_train, e_obs_val, e_obs_test = e_obs[training_index], e_obs[val_index], e_obs[test_index]
    print(e_obs_train.shape)
    print(e_obs_test.shape)
 #   e_hat = np.asarray(e_hat)
 #   e_obs = np.asarray(data['e_obs'])
    e_true_obs = np.asarray(data['e_true_obs'])
    e_true_obs = torch.from_numpy(np.asarray(e_true_obs, dtype=np.float32).reshape(-1, 1))

    e_true_obs_train, e_true_obs_val, e_true_obs_test = e_true_obs[training_index], e_true_obs[val_index], e_true_obs[test_index]


    print(X_train.shape)
    print(X_test.shape)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()

    running_loss = 0
    epochs = 10000
    patience_counter = 0
    patience = 30
    alpha = 0
    best_val_loss = float('inf')
    for i in range(epochs):
        optimizer.zero_grad()

        r_hat_train = model(X_train)

        loss = loss_fn(r_hat_train, y_train)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if i % 50 == 0:
            last_loss = running_loss / 50
            print(last_loss)
            running_loss = 0

        if i % 100 == 0:
            model.eval()
            with torch.no_grad():
                r_hat_val = model(X_val)
                best_alpha_val_loss = float('inf')
                for a in [0.05, 0.1, 0.25, 0.5, 1.0]:
                    e_hat_val = torch.clamp(e_obs_val + a * r_hat_val, 0, 1)
                    val_loss = ((e_hat_val - e_true_obs_val)**2).mean()
                    if (val_loss < best_alpha_val_loss):
                        best_alpha_val_loss = val_loss
                        alpha = a

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(model)
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter > patience:
                    model = best_state
                    break

            model.train()


    e_hat_train = torch.clamp(e_obs_train + r_hat_train, 0, 1)

    mse_obs_train = torch.mean((e_obs_train - e_true_obs_train)**2)
    print(f"mse observed pre eval is = {mse_obs_train}")
    mse_residual_train = torch.mean((e_hat_train - e_true_obs_train)**2)
    print(f"mse residual pre eval = {mse_residual_train}")
    
    model.eval()
    with torch.no_grad():
        r_hat_test = model(X_test)
        e_hat_test = torch.clamp(e_obs_test + alpha * r_hat_test, 0, 1)
    
    clamp_rate = 0
    with torch.no_grad():
        e_raw_test = e_obs_test + r_hat_test
        clamp_rate = ((e_raw_test < 0.0) | (e_raw_test > 1.0)).float().mean().item()

    print(f"clamp rate = {clamp_rate}")

    mse_obs_test = torch.mean((e_obs_test - e_true_obs_test)**2)
    print(f"mse observed post eval is = {mse_obs_test}")
    mse_residual_test = torch.mean((e_hat_test - e_true_obs_test)**2)
    print(f"mse residual post eval = {mse_residual_test}")

    r_true_np_test = (e_true_obs_test - e_obs_test).cpu().numpy()
    r_hat_np_test = r_hat_test.cpu().numpy()
    print(r_true_np_test.mean(), r_true_np_test.std())
    print(r_hat_np_test.mean(), r_hat_np_test.std())