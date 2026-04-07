import torch as th
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class KNN:

    def __init__(self, nb_features, nb_classes, data, k, weighted=False):
        self.nb_features = nb_features
        self.nb_classes = nb_classes
        self.k = k
        self.weighted = weighted
        self.X = th.tensor(data['x'], dtype=th.float32)
        self.Y = th.tensor(data['y'], dtype=th.int64)

    def predict(self, query_data, batch_size=512):
        Xq = th.tensor(query_data['x'], dtype=th.float32)
        predictions = []

        for start in range(0, Xq.shape[0], batch_size):
            end = start + batch_size
            batch = Xq[start:end]

            dists = th.cdist(batch, self.X)
            dists_k, idxs = th.topk(dists, self.k, largest=False)
            classes = self.Y[idxs]

            if self.weighted:
                w = 1 / (dists_k + 1e-8)
            else:
                w = th.ones_like(dists_k) / self.k

            classes_one_hot = th.nn.functional.one_hot(classes, self.nb_classes).float()
            scores = th.sum(w.unsqueeze(-1) * classes_one_hot, dim=1)
            batch_predictions = th.argmax(scores, dim=1)

            predictions.append(batch_predictions)

        predictions = th.cat(predictions)

        accuracy = None

        if query_data['y'] is not None:
            actual = th.tensor(query_data['y'], dtype=th.int64)
            accuracy = (predictions == actual).float().mean().item()

        return predictions.numpy(), accuracy


np.random.seed(7)
th.manual_seed(7)

all_x = np.loadtxt(
    'data/spaceship-titanic.csv',
    delimiter=',',
    skiprows=1,
    usecols=(7, 8),
    dtype=str
)

all_y = np.loadtxt(
    'data/spaceship-titanic.csv',
    delimiter=',',
    skiprows=1,
    usecols=(13,),
    dtype=str
)

all_x[all_x == ''] = 'nan'
x_data = all_x.astype(np.float32)

y_data = (all_y == 'True').astype(np.int64)

data = dict()
data['x'] = x_data
data['y'] = y_data

nb_samples = len(data['y'])
indices = np.random.permutation(nb_samples)
train_size = int(0.8 * nb_samples)

train_indices = indices[:train_size]
test_indices = indices[train_size:]

train_x = data['x'][train_indices].copy()
test_x = data['x'][test_indices].copy()
train_y = data['y'][train_indices]
test_y = data['y'][test_indices]

# NaN vrednosti popunjavamo srednjom vrednoscu iz trening skupa
fill_values = np.nanmean(train_x, axis=0)

train_nan_rows, train_nan_cols = np.where(np.isnan(train_x))
test_nan_rows, test_nan_cols = np.where(np.isnan(test_x))

train_x[train_nan_rows, train_nan_cols] = fill_values[train_nan_cols]
test_x[test_nan_rows, test_nan_cols] = fill_values[test_nan_cols]

# Standardizacija
x_mean = np.mean(train_x, axis=0)
x_std = np.std(train_x, axis=0)
x_std[x_std == 0] = 1.0

train_x = ((train_x - x_mean) / x_std).astype(np.float32)
test_x = ((test_x - x_mean) / x_std).astype(np.float32)

train_data = {'x': train_x, 'y': train_y}
test_data = {'x': test_x, 'y': test_y}

nb_features = 2
nb_classes = 2

k_values = np.arange(1, 51)
accuracies = []

for k in k_values:
    knn = KNN(nb_features, nb_classes, train_data, k, weighted=False)
    _, accuracy = knn.predict(test_data)
    accuracies.append(accuracy)
    print(f'k = {k:2d}, accuracy = {accuracy:.5f}')

accuracies = np.array(accuracies)

best_index = np.argmax(accuracies)
best_k = k_values[best_index]
best_accuracy = accuracies[best_index]

print(f'\nNajbolje k je: {best_k}')
print(f'Najveci accuracy je: {best_accuracy:.5f}')

plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o')
plt.scatter(best_k, best_accuracy, s=80, label=f'Najbolje k = {best_k}')
plt.xlabel('k')
plt.ylabel('Accuracy na test skupu')
plt.title('Zavisnost accuracy metrike od parametra k')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


"""
Diskusija:

Za male vrednosti parametra k model je osetljiviji na pojedinacne primere i sum u podacima,
pa accuracy moze vise da osciluje. Kako k raste, odluka zavisi od veceg broja suseda i model
postaje stabilniji.

Najbolji izbor je ona vrednost k za koju accuracy na test skupu dostize maksimum. Program tu
vrednost automatski ispisuje. Uopsteno, suvise malo k moze da vodi ka overfitting-u, dok
suvise veliko k moze da vodi ka underfitting-u, jer granica odlucivanja postaje previse gruba.
"""