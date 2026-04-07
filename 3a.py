import torch as th
import numpy as np
import pandas as pd
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
        #novi podaci za koje hocemo predikciju
        Xq = th.tensor(query_data['x'], dtype=th.float32)
        #skupljamo predikcije u delovima (batch-evima) da ne bih imao problema sa memorijom
        predictions = []

        #prolazimo kroz upitne podatke u delovima, racunamo udaljenosti do svih trening primera, nalazimo k najblizih, i na osnovu njihovih klasa pravimo predikciju
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

all_data = pd.read_csv('data/spaceship-titanic.csv')

selected_features = ['RoomService', 'FoodCourt']
target_name = 'Transported'

data = dict()
data['x'] = all_data[selected_features].copy()
data['y'] = all_data[target_name].astype(int).to_numpy()

nb_samples = len(data['y'])
indices = np.random.permutation(nb_samples)
train_size = int(0.8 * nb_samples)

train_indices = indices[:train_size]
test_indices = indices[train_size:]

train_x = data['x'].iloc[train_indices].copy()
test_x = data['x'].iloc[test_indices].copy()
train_y = data['y'][train_indices]
test_y = data['y'][test_indices]

# NaN vrednosti popunjavamo srednjom vrednoscu iz trening skupa.
fill_values = train_x.mean()
train_x = train_x.fillna(fill_values)
test_x = test_x.fillna(fill_values)

# k-NN zavisi od skale feature-a, pa ih standardizujemo.
x_mean = train_x.mean()
x_std = train_x.std()
x_std[x_std == 0] = 1.0

train_x = ((train_x - x_mean) / x_std).to_numpy(dtype='float32')
test_x = ((test_x - x_mean) / x_std).to_numpy(dtype='float32')

train_data = {'x': train_x, 'y': train_y}
test_data = {'x': test_x, 'y': test_y}

nb_features = 2
nb_classes = 2
k = 15

knn = KNN(nb_features, nb_classes, train_data, k, weighted=False)
test_predictions, accuracy = knn.predict(test_data)

print(f'Test set accuracy for k={k}: {accuracy:.5f}')


x_min, x_max = train_x[:, 0].min() - 0.5, train_x[:, 0].max() + 0.5
y_min, y_max = train_x[:, 1].min() - 0.5, train_x[:, 1].max() + 0.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 250, dtype='float32'),
    np.linspace(y_min, y_max, 250, dtype='float32')
)

grid_points = np.column_stack((xx.ravel(), yy.ravel())).astype('float32')
grid_predictions, _ = knn.predict({'x': grid_points, 'y': None}, batch_size=1024)
grid_predictions = grid_predictions.reshape(xx.shape)

background_cmap = ListedColormap(['#bde0fe', '#ffd6a5'])
point_cmap = ListedColormap(['#1d3557', '#d00000'])

plt.figure(figsize=(10, 7))
plt.contourf(xx, yy, grid_predictions, alpha=0.45, cmap=background_cmap)
plt.scatter(
    train_x[:, 0],
    train_x[:, 1],
    c=train_y,
    cmap=point_cmap,
    edgecolors='black',
    s=35,
    alpha=0.8,
)

plt.xlabel('RoomService (standardized)')
plt.ylabel('FoodCourt (standardized)')
plt.title(f'k-NN klasifikacija za k={k}')
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()
