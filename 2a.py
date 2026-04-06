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


def train_polynomial_model(train_data, nb_features, learning_rate, nb_epochs):
    model = LinearModel(nb_features)
    loss_fn = nn.MSELoss()
    optimizer = th.optim.Adam(params=model.parameters(), lr=learning_rate)

    for epoch in range(nb_epochs):
        model.train()
        y_pred = model(train_data['x']).squeeze()
        loss = loss_fn(y_pred, train_data['y'])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 500 == 0:
            print(f"Stepen: {nb_features} | Epoch: {epoch + 1}/{nb_epochs} | Avg loss: {loss.item():.5f}")

    model.eval()

    with th.inference_mode():
        final_loss = loss_fn(model(train_data['x']).squeeze(), train_data['y']).item()

    return model, final_loss


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

learning_rate = 0.01
nb_epochs = 3000
degrees = range(1, 7)

final_losses = []
curve_predictions = dict() 

x_plot_original = np.linspace(np.min(x_original), np.max(x_original), 200, dtype='float32')
x_plot_normalized = (x_plot_original - x_mean) / x_std

colors = cm.viridis(np.linspace(0.1, 0.9, len(list(degrees))))

for color, degree in zip(colors, degrees):
    train_x = create_feature_matrix(data['x'], nb_features=degree)
    train_data = dict()
    train_data['x'] = th.tensor(train_x, dtype=th.float32)
    train_data['y'] = th.tensor(data['y'], dtype=th.float32)

    model, final_loss = train_polynomial_model(
        train_data=train_data,
        nb_features=degree,
        learning_rate=learning_rate,
        nb_epochs=nb_epochs,
    )

    final_losses.append(final_loss)

    xs = create_feature_matrix(x_plot_normalized, nb_features=degree)

    with th.inference_mode():
        ys_normalized = model(th.tensor(xs, dtype=th.float32)).squeeze().numpy()

    ys_original = ys_normalized * y_std + y_mean #vratimo u orginalnu skalu pre crtanja
    curve_predictions[degree] = dict(color=color, y=ys_original)

    print(f"Stepen polinoma: {degree} | Finalni MSE na celom skupu: {final_loss:.5f}")


plt.figure(figsize=(10, 6))
plt.scatter(x_original, y_original, color='black', alpha=0.65, label='Podaci')

for degree in degrees:
    plt.plot(
        x_plot_original,
        curve_predictions[degree]['y'],
        color=curve_predictions[degree]['color'],
        linewidth=2,
        label=f'Stepen {degree}',
    )

plt.xlabel('x')
plt.ylabel('y')
plt.title('Polinomijalna regresija za stepene 1-6')
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

plt.figure(figsize=(8, 5))
plt.plot(list(degrees), final_losses, marker='o', linewidth=2, color='tab:red')
plt.xticks(list(degrees))
plt.xlabel('Stepen polinoma')
plt.ylabel('Finalni MSE na celom skupu')
plt.title('Zavisnost funkcije troska od stepena polinoma')
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()


# Komentar:
# Stepeni 1 i 2 ocigledno underfituju skup, jer su njihove krive previse jednostavne i
# finalni MSE je znatno veci. Najveci dobitak dobijamo kod stepena 3, sto znaci da skup
# ima izrazenu kubnu komponentu. Za stepene 4, 5 i 6 trosak se jos neznatno smanjuje, ali
# je to poboljsanje vrlo malo, pa vidimo "diminishing returns" - slozeniji model ne donosi
# bitno bolji fit na celom skupu. Dakle, na osnovu ovog grafika razuman izbor bio bi stepen
# 3 ili 4, jer visi stepeni povecavaju slozenost modela bez znacajnog dodatnog dobitka.
