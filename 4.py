import csv
import re
from collections import Counter
from html import unescape

import numpy as np
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS


class MultinomialNaiveBayes:
    def __init__(self, nb_classes, nb_words, pseudocount=1.0):
        self.nb_classes = nb_classes
        self.nb_words = nb_words
        self.pseudocount = pseudocount
        self.priors = None
        self.like = None

    def fit(self, X, Y):
        nb_examples = X.shape[0]

        self.priors = np.bincount(Y, minlength=self.nb_classes) / nb_examples

        occs = np.zeros((self.nb_classes, self.nb_words), dtype=np.float64)
        for c in range(self.nb_classes):
            occs[c] = np.asarray(X[Y == c].sum(axis=0)).ravel()

        self.like = np.zeros((self.nb_classes, self.nb_words), dtype=np.float64)
        for c in range(self.nb_classes):
            up = occs[c] + self.pseudocount
            down = np.sum(occs[c]) + self.nb_words * self.pseudocount
            self.like[c] = up / down

    def predict(self, X):
        log_priors = np.log(self.priors + 1e-12)
        log_like = np.log(self.like + 1e-12)
        scores = X @ log_like.T + log_priors
        predictions = np.asarray(np.argmax(scores, axis=1)).ravel()
        return predictions

    def predict_and_accuracy(self, X, Y=None):
        predictions = self.predict(X)

        accuracy = None
        if Y is not None:
            accuracy = np.mean(predictions == Y)

        return predictions, accuracy


stemmer = PorterStemmer()

base_stopwords = set(ENGLISH_STOP_WORDS)
base_stopwords = base_stopwords.difference({'no', 'not', 'nor', 'never'})
base_stopwords = base_stopwords.union({
    'amp', 'rt', 'im', 'ive', 'dont', 'didnt', 'doesnt',
    'cant', 'couldnt', 'wont', 'wouldnt', 'u', 'ur'
})
stemmed_stopwords = {stemmer.stem(word) for word in base_stopwords}


def clean_label(label):
    label = str(label).strip().lower()

    if label == 'extremely positive':
        return 'Positive'
    if label == 'positive':
        return 'Positive'
    if label == 'extremely negative':
        return 'Negative'
    if label == 'negative':
        return 'Negative'

    return 'Neutral'


def clean_text(text):
    text = str(text)
    text = unescape(text)
    text = text.lower()

    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'\brt\b', ' ', text)

    text = re.sub(r'covid[-_\s]*19', ' covid ', text)
    text = re.sub(r'covid19', ' covid ', text)
    text = re.sub(r'corona virus', ' coronavirus ', text)

    tokens = re.findall(r"[a-z']+", text)

    cleaned_tokens = []
    for token in tokens:
        if token in {"'", "''"}:
            continue

        if len(token) == 1 and token not in {'a', 'i'}:
            continue

        stemmed = stemmer.stem(token)

        if stemmed in stemmed_stopwords:
            continue

        if len(stemmed) < 2 and stemmed not in {'a', 'i', 'no'}:
            continue

        cleaned_tokens.append(stemmed)

    return ' '.join(cleaned_tokens)


def load_csv_data(path):
    texts = []
    labels = []

    with open(path, 'r', encoding='latin1', newline='') as f:
        reader = csv.DictReader(f)

        for row in reader:
            text = row.get('OriginalTweet', '')
            label = row.get('Sentiment', '')

            texts.append(text if text is not None else '')
            labels.append(label if label is not None else '')

    return np.array(texts, dtype=object), np.array(labels, dtype=object)


def print_top_words(title, counter, top_n=5):
    print(title)
    for word, count in counter.most_common(top_n):
        print(f'  {word:<20} {count}')
    print()


def build_lr_lists(pos_counter, neg_counter, min_count=10, top_n=5):
    lr_values = []

    all_words = set(pos_counter.keys()).intersection(set(neg_counter.keys()))
    for word in all_words:
        if pos_counter[word] >= min_count and neg_counter[word] >= min_count:
            lr = pos_counter[word] / neg_counter[word]
            lr_values.append((word, lr, pos_counter[word], neg_counter[word]))

    highest = sorted(lr_values, key=lambda x: (-x[1], x[0]))[:top_n]
    lowest = sorted(lr_values, key=lambda x: (x[1], x[0]))[:top_n]

    return highest, lowest


def print_lr_words(title, items):
    print(title)
    for word, lr, pos_count, neg_count in items:
        print(f'  {word:<20} LR={lr:.4f} | pos={pos_count} | neg={neg_count}')
    print()


def main():
    train_text_raw, train_labels_raw = load_csv_data('data/Corona_NLP_train.csv')
    test_text_raw, test_labels_raw = load_csv_data('data/Corona_NLP_test.csv')

    train_labels_clean = np.array([clean_label(label) for label in train_labels_raw], dtype=object)
    test_labels_clean = np.array([clean_label(label) for label in test_labels_raw], dtype=object)

    label_to_index = {
        'Negative': 0,
        'Neutral': 1,
        'Positive': 2,
    }
    index_to_label = {value: key for key, value in label_to_index.items()}

    train_y = np.array([label_to_index[label] for label in train_labels_clean], dtype=np.int64)
    test_y = np.array([label_to_index[label] for label in test_labels_clean], dtype=np.int64)

    train_text_clean = [clean_text(text) for text in train_text_raw]
    test_text_clean = [clean_text(text) for text in test_text_raw]

    print('Broj trening primera:', len(train_text_clean))
    print('Broj test primera:', len(test_text_clean))
    print('Klase nakon ciscenja label-a:', sorted(label_to_index.keys()))
    print()

    vectorizer = CountVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2
    )

    X_train = vectorizer.fit_transform(train_text_clean)
    X_test = vectorizer.transform(test_text_clean)

    nb_classes = len(label_to_index)
    nb_words = X_train.shape[1]
    alpha = 0.5

    print('Broj reci / n-grama u vokabularu:', nb_words)
    print('Pseudocount (alpha):', alpha)
    print()

    accuracies = []
    final_predictions = None

    for run in range(3):
        model = MultinomialNaiveBayes(
            nb_classes=nb_classes,
            nb_words=nb_words,
            pseudocount=alpha
        )
        model.fit(X_train, train_y)
        predictions, accuracy = model.predict_and_accuracy(X_test, test_y)

        accuracies.append(accuracy)
        final_predictions = predictions
        print(f'Pokretanje {run + 1}: accuracy = {accuracy:.5f}')

    accuracies = np.array(accuracies)
    mean_accuracy = np.mean(accuracies)

    print()
    print(f'Prosecna accuracy kroz 3 pokretanja: {mean_accuracy:.5f}')
    print()

    predicted_labels = [index_to_label[idx] for idx in final_predictions[:10]]
    print('Prvih 10 predikcija na test skupu:')
    print(predicted_labels)
    print()

    positive_texts = [
        text for text, label in zip(train_text_clean, train_labels_clean)
        if label == 'Positive'
    ]
    negative_texts = [
        text for text, label in zip(train_text_clean, train_labels_clean)
        if label == 'Negative'
    ]

    positive_counter = Counter()
    negative_counter = Counter()

    for text in positive_texts:
        positive_counter.update(text.split())

    for text in negative_texts:
        negative_counter.update(text.split())

    print_top_words('5 najcescih reci u pozitivnim tvitovima:', positive_counter, top_n=5)
    print_top_words('5 najcescih reci u negativnim tvitovima:', negative_counter, top_n=5)

    highest_lr, lowest_lr = build_lr_lists(
        positive_counter,
        negative_counter,
        min_count=10,
        top_n=5
    )

    print_lr_words('5 reci sa najvecom LR vrednoscu:', highest_lr)
    print_lr_words('5 reci sa najmanjom LR vrednoscu:', lowest_lr)


if __name__ == '__main__':
    main()


"""
Diskusija:

1) Ciscenje podataka

Najpre su label-e ociscene tako sto su vrednosti "Extremely Positive" i "Positive"
spojene u klasu Positive, a "Extremely Negative" i "Negative" u klasu Negative.
Neutral je zadrzan kao posebna klasa. Time se smanjuje sum u label-ama i problem
postaje pogodniji za Naive Bayes klasifikaciju.

Tekstovi su ocisceni uklanjanjem URL-ova, mention-a, RT oznaka i nepotrebnih simbola,
a zatim su tokenizovani regex-om, spusteni na mala slova i stemovani. Uz to su uklonjene
stop reci, ali su negacije kao "not", "no" i "never" zadrzane jer mogu biti veoma vazne
za sentiment.

2) Feature vektori i model

Za feature-e je koriscen Bag of Words pristup uz unigram-e i bigram-e, sa ogranicenjem
vokabulara na najvise 10000 feature-a. Ovo je dobar kompromis izmedju informativnosti
i dimenzionalnosti feature prostora. Model je Multinomial Naive Bayes sa Laplace-ovim
izravnavanjem.

3) Najcesce reci u pozitivnim i negativnim tvitovima

Najcesce reci u pozitivnim tvitovima cesto ukljucuju reci koje opisuju podrsku,
zahvalnost, nadu ili sigurnost. U negativnim tvitovima se cesto pojavljuju reci
povezane sa strahom, nedostatkom, problemima i ogranicenjima. Sama frekvencija
reci nije dovoljna da bi se procenilo koliko je rec karakteristicna za jednu klasu,
jer neke vrlo ceste reci mogu biti ceste u obe klase.

4) Znacenje metrike LR

LR(rec) = broj_pojavljivanja_u_pozitivnim_tvitovima / broj_pojavljivanja_u_negativnim_tvitovima

Ako je LR mnogo vece od 1, rec je tipicnija za pozitivne tvitove.
Ako je LR mnogo manje od 1, rec je tipicnija za negativne tvitove.
Ako je LR blizu 1, rec se pojavljuje slicno u obe klase.

Zato LR bolje pokazuje diskriminativnost reci nego sama frekvencija.

5) Poredjenje sa najcescim recima

Najcesce reci pokazuju koje se teme cesto spominju u korpusu, dok LR pokazuje koje
reci najvise naginju jednoj klasi u odnosu na drugu. Zato reci sa najvecom i najmanjom
LR vrednoscu cesto bolje objasnjavaju zasto model pravi odredjene klasifikacije nego
same najfrekventnije reci.
"""