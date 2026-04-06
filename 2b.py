import matplotlib
matplotlib.use('TkAgg')

import torch as th
import numpy as np
from torch import nn
import matplotlib.pyplot as plt


def create_feature_matrix(x, nb_features):
    tmp_features = []

    for deg in range(1, nb_features + 1):
        tmp_features.append(np.power(x, deg))

    return np.column_stack(tmp_features)


class LinearModel(nn.Module):
    def __init__(self, nb_features):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=nb_features, out_features=1)

    def forward(self, x):
        return self.linear_layer(x)

    def compute_l1_loss(self, lmbd):
        params = []

        for p in self.parameters():
            params.append(p.view(-1))

        w = th.cat(params)
        return lmbd * th.abs(w).sum()

    def compute_l2_loss(self, lmbd):
        params = []

        for p in self.parameters():
            params.append(p.view(-1))

        w = th.cat(params)
        return lmbd * th.pow(w, 2).sum()


def train_model(x, y, nb_features, learning_rate, nb_epochs, lmbd):
    th.manual_seed(7)

    model = LinearModel(nb_features)
    loss_fn = nn.MSELoss()
    optimizer = th.optim.Adam(params=model.parameters(), lr=learning_rate)

    for epoch in range(nb_epochs):
        model.train()

        y_pred = model(x).squeeze()
        data_loss = loss_fn(y_pred, y)
        reg_loss = model.compute_l2_loss(lmbd)
        loss = data_loss + reg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(
                f"lambda = {lmbd:<7} | "
                f"Epoch: {epoch + 1}/{nb_epochs} | "
                f"Data loss: {data_loss.item():.5f} | "
                f"Total loss: {loss.item():.5f}"
            )

    model.eval()
    with th.inference_mode():
        final_pred = model(x).squeeze()
        final_data_loss = loss_fn(final_pred, y).item()
        final_total_loss = (loss_fn(final_pred, y) + model.compute_l2_loss(lmbd)).item()

    return model, final_data_loss, final_total_loss


np.random.seed(7)
th.manual_seed(7)

all_data = np.loadtxt('data/funky.csv', delimiter=',', dtype='float32')

data = dict()
data['x'] = all_data[:, 0]
data['y'] = all_data[:, 1]

nb_samples = data['x'].shape[0]

indices = np.random.permutation(nb_samples)
data['x'] = data['x'][indices]
data['y'] = data['y'][indices]

x_original = data['x'].copy()
y_original = data['y'].copy()

x_mean = np.mean(data['x'])
x_std = np.std(data['x'])
y_mean = np.mean(data['y'])
y_std = np.std(data['y'])

data['x'] = (data['x'] - x_mean) / x_std
data['y'] = (data['y'] - y_mean) / y_std

nb_features = 3
data['x'] = create_feature_matrix(data['x'], nb_features=nb_features)

data['x'] = th.tensor(data['x'], dtype=th.float32)
data['y'] = th.tensor(data['y'], dtype=th.float32)

learning_rate = 0.01
nb_epochs = 1000
lambdas = [0, 0.001, 0.01, 0.1, 1, 10, 100]

models = []
final_data_losses = []
final_total_losses = []

for lmbd in lambdas:
    print(f"\n--- Trening za lambda = {lmbd} ---")

    model, final_data_loss, final_total_loss = train_model(
        x=data['x'],
        y=data['y'],
        nb_features=nb_features,
        learning_rate=learning_rate,
        nb_epochs=nb_epochs,
        lmbd=lmbd
    )

    models.append(model)
    final_data_losses.append(final_data_loss)
    final_total_losses.append(final_total_loss)

    print(f"Final data loss  (MSE na celom skupu): {final_data_loss:.5f}")
    print(f"Final total loss (MSE + L2):          {final_total_loss:.5f}")

plt.figure(figsize=(12, 7))
plt.scatter(x_original, y_original, s=25, alpha=0.7, label='Podaci')

x_plot_original = np.linspace(np.min(x_original), np.max(x_original), 400, dtype='float32')
x_plot_normalized = (x_plot_original - x_mean) / x_std
x_plot_features = create_feature_matrix(x_plot_normalized, nb_features=nb_features)
x_plot_tensor = th.tensor(x_plot_features, dtype=th.float32)

for model, lmbd in zip(models, lambdas):
    model.eval()
    with th.inference_mode():
        y_plot_normalized = model(x_plot_tensor).squeeze().numpy()

    y_plot_original = y_plot_normalized * y_std + y_mean
    plt.plot(x_plot_original, y_plot_original, linewidth=2, label=f'lambda = {lmbd}')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Polinomijalna regresija stepena 3 sa L2 regularizacijom')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
x_axis = np.arange(len(lambdas))

plt.plot(x_axis, final_total_losses, marker='o')
plt.xticks(x_axis, [str(l) for l in lambdas])

plt.xlabel('lambda')
plt.ylabel('Finalna funkcija troška na celom skupu')
plt.title('Zavisnost finalne funkcije troška od parametra lambda')
plt.grid(True)
plt.tight_layout()
plt.show()