import torch as th
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from matplotlib import cm


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

    def compute_l2_loss(self, lmbd):
        params = []

        for p in self.parameters():
            params.append(p.view(-1))

        w = th.cat(params)
        return lmbd * th.pow(w, 2).sum()


def train_polynomial_model(train_data, nb_features, learning_rate, nb_epochs, lmbd):
    th.manual_seed(7)

    model = LinearModel(nb_features)
    loss_fn = nn.MSELoss()
    optimizer = th.optim.Adam(params=model.parameters(), lr=learning_rate)

    for epoch in range(nb_epochs):
        model.train()

        y_pred = model(train_data['x']).squeeze()
        data_loss = loss_fn(y_pred, train_data['y'])
        reg_loss = model.compute_l2_loss(lmbd)
        loss = data_loss + reg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 500 == 0:
            print(
                f"lambda: {lmbd:<7} | "
                f"Epoch: {epoch + 1}/{nb_epochs} | "
                f"MSE: {data_loss.item():.5f} | "
                f"Ukupan loss: {loss.item():.5f}"
            )

    model.eval()

    with th.inference_mode():
        final_pred = model(train_data['x']).squeeze()
        final_data_loss = loss_fn(final_pred, train_data['y']).item()
        final_total_loss = (loss_fn(final_pred, train_data['y']) + model.compute_l2_loss(lmbd)).item()

    return model, final_data_loss, final_total_loss


th.manual_seed(7)
np.random.seed(7)

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
learning_rate = 0.01
nb_epochs = 3000
lambdas = [0, 0.001, 0.01, 0.1, 1, 10, 100]

train_x = create_feature_matrix(data['x'], nb_features=nb_features)
train_data = dict()
train_data['x'] = th.tensor(train_x, dtype=th.float32)
train_data['y'] = th.tensor(data['y'], dtype=th.float32)

models = []
final_data_losses = []
final_total_losses = []
curve_predictions = dict()

x_plot_original = np.linspace(np.min(x_original), np.max(x_original), 200, dtype='float32')
x_plot_normalized = (x_plot_original - x_mean) / x_std
x_plot_features = create_feature_matrix(x_plot_normalized, nb_features=nb_features)
x_plot_tensor = th.tensor(x_plot_features, dtype=th.float32)

colors = cm.plasma(np.linspace(0.1, 0.9, len(lambdas)))

for color, lmbd in zip(colors, lambdas):
    model, final_data_loss, final_total_loss = train_polynomial_model(
        train_data=train_data,
        nb_features=nb_features,
        learning_rate=learning_rate,
        nb_epochs=nb_epochs,
        lmbd=lmbd,
    )

    models.append(model)
    final_data_losses.append(final_data_loss)
    final_total_losses.append(final_total_loss)

    with th.inference_mode():
        ys_normalized = model(x_plot_tensor).squeeze().numpy()

    ys_original = ys_normalized * y_std + y_mean
    curve_predictions[lmbd] = dict(color=color, y=ys_original)

    print(f"lambda = {lmbd:<7} | Finalni MSE na celom skupu: {final_data_loss:.5f}")
    print(f"lambda = {lmbd:<7} | Finalni ukupan loss na celom skupu: {final_total_loss:.5f}")


plt.figure(figsize=(10, 6))
plt.scatter(x_original, y_original, color='black', alpha=0.65, label='Podaci')

for lmbd in lambdas:
    plt.plot(
        x_plot_original,
        curve_predictions[lmbd]['y'],
        color=curve_predictions[lmbd]['color'],
        linewidth=2,
        label=f'lambda = {lmbd}',
    )

plt.xlabel('x')
plt.ylabel('y')
plt.title('Polinomijalna regresija stepena 3 sa L2 regularizacijom')
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

plt.figure(figsize=(8, 5))
x_axis = np.arange(len(lambdas))
plt.plot(x_axis, final_total_losses, marker='o', linewidth=2, color='tab:blue')
plt.xticks(x_axis, [str(lmbd) for lmbd in lambdas])
plt.xlabel('lambda')
plt.ylabel('Finalni ukupan loss na celom skupu')
plt.title('Zavisnost funkcije troska od parametra lambda')
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()


# Komentar:
# Kada je lambda mala, regularizacija je slaba i model zadrzava fleksibilnost kubnog
# polinoma, pa dobro prati oblik podataka. Kako lambda raste, L2 kazna sve jace gura
# tezine ka manjim vrednostima, pa kriva postaje sve "mirnija" i manje prilagodjena
# podacima. Za vrlo velike vrednosti lambda (posebno 10 i 100) model postaje previse
# ogranicen, pa dolazi do underfitting-a i ukupan trosak raste. Dakle, umerena
# regularizacija moze da stabilizuje model, ali prejaka regularizacija kvari fit.
