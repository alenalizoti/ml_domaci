import torch as th
import numpy as np
import matplotlib.pyplot as plt


class KNN:

    def __init__(self, nb_features, nb_classes, data, k, weighted=False):
        self.nb_features = nb_features
        self.nb_classes = nb_classes
        self.k = k
        self.weighted = weighted
        self.X = th.tensor(data['x'], dtype=th.float32)
        self.Y = th.tensor(data['y'], dtype=th.int64)

    def predict(self, query_data, batch_size=512):
        # novi podaci za koje hocemo predikciju
        Xq = th.tensor(query_data['x'], dtype=th.float32)
        predictions = []

        # prolazimo kroz upitne podatke u delovima da ne trosimo previse memorije
        for start in range(0, Xq.shape[0], batch_size):
            end = start + batch_size
            batch = Xq[start:end]

            # racunamo udaljenosti do svih trening primera i uzimamo k najblizih
            dists = th.cdist(batch, self.X)
            dists_k, idxs = th.topk(dists, self.k, largest=False)
            classes = self.Y[idxs]

            if self.weighted:
                w = 1 / (dists_k + 1e-8)
            else:
                w = th.ones_like(dists_k) / self.k

            # svaki sused glasa za svoju klasu
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


def parse_float_column(col):
    values = np.full(col.shape[0], np.nan, dtype=np.float32)

    for i, value in enumerate(col):
        if value != '':
            values[i] = np.float32(value)

    return values


def parse_bool_column(col):
    values = np.full(col.shape[0], np.nan, dtype=np.float32)

    for i, value in enumerate(col):
        if value == 'True':
            values[i] = 1.0
        elif value == 'False':
            values[i] = 0.0

    return values


def binary_mode(values):
    valid_values = values[~np.isnan(values)]

    if valid_values.shape[0] == 0:
        return 0.0

    ones = np.sum(valid_values == 1.0)
    zeros = np.sum(valid_values == 0.0)

    if ones >= zeros:
        return 1.0
    else:
        return 0.0


def parse_passenger_id(col):
    group_ids = np.array(['Unknown'] * col.shape[0], dtype=object)
    passenger_numbers = np.full(col.shape[0], np.nan, dtype=np.float32)

    for i, value in enumerate(col):
        if value != '' and '_' in value:
            parts = value.split('_')

            if len(parts) == 2:
                group_ids[i] = parts[0]
                passenger_numbers[i] = np.float32(parts[1])

    return group_ids, passenger_numbers


def compute_count_feature(train_labels, test_labels):
    valid_train = train_labels[train_labels != 'Unknown']

    if valid_train.shape[0] == 0:
        train_counts = np.ones(train_labels.shape[0], dtype=np.float32)
        test_counts = np.ones(test_labels.shape[0], dtype=np.float32)
        return train_counts, test_counts

    unique_labels, counts = np.unique(valid_train, return_counts=True)
    count_map = dict(zip(unique_labels, counts))

    train_counts = np.array(
        [count_map[label] if label in count_map else 1 for label in train_labels],
        dtype=np.float32
    )

    test_counts = np.array(
        [count_map[label] if label in count_map else 1 for label in test_labels],
        dtype=np.float32
    )

    return train_counts, test_counts


def parse_cabin(col):
    decks = np.array(['Unknown'] * col.shape[0], dtype=object)
    cabin_numbers = np.full(col.shape[0], np.nan, dtype=np.float32)
    sides = np.array(['Unknown'] * col.shape[0], dtype=object)

    for i, value in enumerate(col):
        if value != '' and value.count('/') == 2:
            parts = value.split('/')

            decks[i] = parts[0] if parts[0] != '' else 'Unknown'
            sides[i] = parts[2] if parts[2] != '' else 'Unknown'

            if parts[1] != '':
                cabin_numbers[i] = np.float32(parts[1])

    return decks, cabin_numbers, sides


def parse_name(col):
    name_lengths = np.zeros(col.shape[0], dtype=np.float32)
    surname_lengths = np.zeros(col.shape[0], dtype=np.float32)
    surnames = np.array(['Unknown'] * col.shape[0], dtype=object)

    for i, value in enumerate(col):
        if value != '':
            name_lengths[i] = len(value)

            parts = value.split()
            if len(parts) > 0:
                surname = parts[-1]
                surnames[i] = surname
                surname_lengths[i] = len(surname)

    return name_lengths, surname_lengths, surnames


def one_hot_encode(train_col, test_col):
    train_filled = np.array(
        [value if value != '' else 'Unknown' for value in train_col],
        dtype=object
    )
    test_filled = np.array(
        [value if value != '' else 'Unknown' for value in test_col],
        dtype=object
    )

    categories = np.unique(np.append(train_filled, 'Unknown'))
    category_to_index = {category: i for i, category in enumerate(categories)}

    train_encoded = np.zeros((train_filled.shape[0], len(categories)), dtype=np.float32)
    test_encoded = np.zeros((test_filled.shape[0], len(categories)), dtype=np.float32)

    unknown_index = category_to_index['Unknown']

    for i, value in enumerate(train_filled):
        idx = category_to_index.get(value, unknown_index)
        train_encoded[i, idx] = 1.0

    for i, value in enumerate(test_filled):
        idx = category_to_index.get(value, unknown_index)
        test_encoded[i, idx] = 1.0

    return train_encoded, test_encoded


np.random.seed(7)
th.manual_seed(7)

# Ucitavamo ceo fajl kao stringove, jer skup sadrzi i numericke i kategoricke kolone.
all_data = np.loadtxt(
    'data/spaceship-titanic.csv',
    delimiter=',',
    skiprows=1,
    dtype=str
)

x_raw = all_data[:, :-1]
y_raw = all_data[:, -1]

y_data = (y_raw == 'True').astype(np.int64)

data = dict()
data['x'] = x_raw
data['y'] = y_data

# Nasumicna podela na trening i test skup.
nb_samples = len(data['y'])
indices = np.random.permutation(nb_samples)
train_size = int(0.8 * nb_samples)

train_indices = indices[:train_size]
test_indices = indices[train_size:]

train_x_raw = data['x'][train_indices].copy()
test_x_raw = data['x'][test_indices].copy()
train_y = data['y'][train_indices]
test_y = data['y'][test_indices]

# -------------------------------------------------------------------
# IZVLACENJE FEATURE-A IZ SVIH ORIGINALNIH KOLONA
# PassengerId -> group size + passenger number
# HomePlanet -> one-hot
# CryoSleep -> 0/1
# Cabin -> deck + cabin number + side
# Destination -> one-hot
# Age -> numericki
# VIP -> 0/1
# RoomService, FoodCourt, ShoppingMall, Spa, VRDeck -> numericki
# Name -> duzina imena + duzina prezimena + broj pojavljivanja prezimena
# -------------------------------------------------------------------

# PassengerId
train_group_ids, train_passenger_numbers = parse_passenger_id(train_x_raw[:, 0])
test_group_ids, test_passenger_numbers = parse_passenger_id(test_x_raw[:, 0])
train_group_sizes, test_group_sizes = compute_count_feature(train_group_ids, test_group_ids)

# HomePlanet
train_homeplanet = train_x_raw[:, 1]
test_homeplanet = test_x_raw[:, 1]

# CryoSleep
train_cryosleep = parse_bool_column(train_x_raw[:, 2])
test_cryosleep = parse_bool_column(test_x_raw[:, 2])

# Cabin
train_cabin_decks, train_cabin_numbers, train_cabin_sides = parse_cabin(train_x_raw[:, 3])
test_cabin_decks, test_cabin_numbers, test_cabin_sides = parse_cabin(test_x_raw[:, 3])

# Destination
train_destination = train_x_raw[:, 4]
test_destination = test_x_raw[:, 4]

# Age
train_age = parse_float_column(train_x_raw[:, 5])
test_age = parse_float_column(test_x_raw[:, 5])

# VIP
train_vip = parse_bool_column(train_x_raw[:, 6])
test_vip = parse_bool_column(test_x_raw[:, 6])

# Numericke troskovne kolone
train_room_service = parse_float_column(train_x_raw[:, 7])
test_room_service = parse_float_column(test_x_raw[:, 7])

train_food_court = parse_float_column(train_x_raw[:, 8])
test_food_court = parse_float_column(test_x_raw[:, 8])

train_shopping_mall = parse_float_column(train_x_raw[:, 9])
test_shopping_mall = parse_float_column(test_x_raw[:, 9])

train_spa = parse_float_column(train_x_raw[:, 10])
test_spa = parse_float_column(test_x_raw[:, 10])

train_vrdeck = parse_float_column(train_x_raw[:, 11])
test_vrdeck = parse_float_column(test_x_raw[:, 11])

# Name
train_name_lengths, train_surname_lengths, train_surnames = parse_name(train_x_raw[:, 12])
test_name_lengths, test_surname_lengths, test_surnames = parse_name(test_x_raw[:, 12])
train_surname_counts, test_surname_counts = compute_count_feature(train_surnames, test_surnames)

# -------------------------------------------------------------------
# NUMERICKE KOLONE: IMPUTACIJA I STANDARDIZACIJA
# -------------------------------------------------------------------

train_numeric = np.column_stack((
    train_passenger_numbers,
    train_group_sizes,
    train_cabin_numbers,
    train_age,
    train_room_service,
    train_food_court,
    train_shopping_mall,
    train_spa,
    train_vrdeck,
    train_name_lengths,
    train_surname_lengths,
    train_surname_counts
)).astype(np.float32)

test_numeric = np.column_stack((
    test_passenger_numbers,
    test_group_sizes,
    test_cabin_numbers,
    test_age,
    test_room_service,
    test_food_court,
    test_shopping_mall,
    test_spa,
    test_vrdeck,
    test_name_lengths,
    test_surname_lengths,
    test_surname_counts
)).astype(np.float32)

numeric_fill_values = np.nanmedian(train_numeric, axis=0)

train_nan_rows, train_nan_cols = np.where(np.isnan(train_numeric))
test_nan_rows, test_nan_cols = np.where(np.isnan(test_numeric))

train_numeric[train_nan_rows, train_nan_cols] = numeric_fill_values[train_nan_cols]
test_numeric[test_nan_rows, test_nan_cols] = numeric_fill_values[test_nan_cols]

x_mean = np.mean(train_numeric, axis=0)
x_std = np.std(train_numeric, axis=0)
x_std[x_std == 0] = 1.0

train_numeric = ((train_numeric - x_mean) / x_std).astype(np.float32)
test_numeric = ((test_numeric - x_mean) / x_std).astype(np.float32)

# -------------------------------------------------------------------
# BINARNE KOLONE
# -------------------------------------------------------------------

train_binary = np.column_stack((train_cryosleep, train_vip)).astype(np.float32)
test_binary = np.column_stack((test_cryosleep, test_vip)).astype(np.float32)

for col in range(train_binary.shape[1]):
    fill_value = binary_mode(train_binary[:, col])
    train_binary[np.isnan(train_binary[:, col]), col] = fill_value
    test_binary[np.isnan(test_binary[:, col]), col] = fill_value

# -------------------------------------------------------------------
# KATEGORICKE KOLONE
# -------------------------------------------------------------------

train_homeplanet_encoded, test_homeplanet_encoded = one_hot_encode(train_homeplanet, test_homeplanet)
train_destination_encoded, test_destination_encoded = one_hot_encode(train_destination, test_destination)
train_cabin_deck_encoded, test_cabin_deck_encoded = one_hot_encode(train_cabin_decks, test_cabin_decks)
train_cabin_side_encoded, test_cabin_side_encoded = one_hot_encode(train_cabin_sides, test_cabin_sides)

# Sve feature-e spajamo u jednu matricu pogodnu za k-NN.
train_x = np.concatenate((
    train_numeric,
    train_binary,
    train_homeplanet_encoded,
    train_destination_encoded,
    train_cabin_deck_encoded,
    train_cabin_side_encoded
), axis=1).astype(np.float32)

test_x = np.concatenate((
    test_numeric,
    test_binary,
    test_homeplanet_encoded,
    test_destination_encoded,
    test_cabin_deck_encoded,
    test_cabin_side_encoded
), axis=1).astype(np.float32)

train_data = {'x': train_x, 'y': train_y}
test_data = {'x': test_x, 'y': test_y}

nb_features = train_x.shape[1]
nb_classes = 2

print(f'Broj feature-a nakon obrade: {nb_features}')

k_values = range(1, 51)
accuracies = []

for k in k_values:
    knn = KNN(nb_features, nb_classes, train_data, k, weighted=False)
    _, accuracy = knn.predict(test_data)
    accuracies.append(accuracy)

    print(f'k = {k:2d} | Test accuracy = {accuracy:.5f}')


best_index = int(np.argmax(accuracies))
best_k = list(k_values)[best_index]
best_accuracy = accuracies[best_index]

print(f'\nNajbolje k na test skupu je: {best_k}')
print(f'Najveci accuracy na test skupu je: {best_accuracy:.5f}')

plt.figure(figsize=(10, 6))
plt.plot(list(k_values), accuracies, marker='o', linewidth=2, color='tab:purple')
plt.scatter(best_k, best_accuracy, color='red', s=80, label=f'Najbolje k = {best_k}')
plt.xlabel('k')
plt.ylabel('Accuracy na test skupu')
plt.title('Zavisnost accuracy metrike od parametra k - svi feature-i')
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.show()


# Komentar:
# U odnosu na zadatak 3b, ovde koristimo mnogo vise informacija o svakom putniku, pa je
# realno ocekivati visi accuracy ili bar stabilniji grafik u srednjem opsegu vrednosti k.
# Dva feature-a iz 3b (RoomService i FoodCourt) nose korisnu informaciju, ali ne opisuju
# dovoljno kompletno svakog putnika. Ukljucivanjem svih kolona dobijamo bogatiju sliku,
# pa k-NN ima vise osnova za trazenje slicnih primera. Na ovoj podeli 3b daje najbolji
# accuracy oko 0.73088, dok 3c dostize oko 0.77688, sto znaci da dodatni feature-i zaista
# poboljsavaju klasifikaciju. Takodje, najbolji rezultat se sada dobija za k = 30, pa se
# vidi da sa bogatijim opisom putnika model moze da koristi i nesto siri lokalni kontekst