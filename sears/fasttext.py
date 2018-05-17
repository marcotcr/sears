
from __future__ import print_function
import numpy as np

import keras
from keras.preprocessing import sequence
import keras.preprocessing.text
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from keras.datasets import imdb
import tempfile


def make_keras_picklable():
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = { 'model_str': model_str }
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name)
        self.__dict__ = model.__dict__


    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__


make_keras_picklable()

def create_ngram_set(input_list, ngram_value=2):
    """
    Extract a set of n-grams from a list of integers.
    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
    {(4, 9), (4, 1), (1, 4), (9, 4)}
    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
    [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    """
    Augment the input list of list (sequences) by appending n-grams values.
    Example: adding bi-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    >>> add_ngram(sequences, token_indice, ngram_range=2)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]
    Example: adding tri-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
    >>> add_ngram(sequences, token_indice, ngram_range=3)
    [[1, 3, 4, 5, 1337], [1, 3, 7, 9, 2, 1337, 2018]]
    """
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for i in range(len(new_list) - ngram_range + 1):
            for ngram_value in range(2, ngram_range + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences

class FastTextClassifier:
    def __init__(self):
        pass

    def predict(self, X):
        x_test = self.tokenizer.texts_to_sequences(X)
        x_test = self.add_ngrams(x_test)
        x_test = sequence.pad_sequences(x_test, maxlen=self.maxlen)
        return self.model.predict_classes(x_test, verbose=0).flatten()
    def predict_proba(self, X):
        x_test = self.tokenizer.texts_to_sequences(X)
        x_test = self.add_ngrams(x_test)
        x_test = sequence.pad_sequences(x_test, maxlen=self.maxlen)
        a = self.model.predict(x_test).flatten()
        a = a.reshape(-1, 1)
        return np.hstack((1 - a, a))
    def fit(self, X, Y, ngram_range=1, max_features=20000, maxlen=400,
            batch_size=32, embedding_dims=50, epochs=5):
        self.tokenizer = keras.preprocessing.text.Tokenizer(
            num_words=max_features, split=" ", char_level=False)
        self.tokenizer.fit_on_texts(X)
        x_train = self.tokenizer.texts_to_sequences(X)
        self.ngram_range = ngram_range
        self.maxlen = maxlen
        self.add_ngrams = lambda x: x
        if ngram_range > 1:
            ngram_set = set()
            for input_list in x_train:
                for i in range(2, ngram_range + 1):
                    set_of_ngram = create_ngram_set(input_list, ngram_value=i)
                    ngram_set.update(set_of_ngram)

            # Dictionary mapping n-gram token to a unique integer.
            # Integer values are greater than max_features in order
            # to avoid collision with existing features.
            start_index = max_features + 1
            self.token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
            indice_token = {self.token_indice[k]: k for k in self.token_indice}

            # max_features is the highest integer that could be found in the dataset.
            max_features = np.max(list(indice_token.keys())) + 1
            self.add_ngrams = lambda x: add_ngram(x, self.token_indice,
                                                  self.ngram_range)
            x_train = self.add_ngrams(x_train)
            print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
        x_train = sequence.pad_sequences(x_train, maxlen=self.maxlen)
        self.model = Sequential()

        # we start off with an efficient embedding layer which maps
        # our vocab indices into embedding_dims dimensions
        self.model.add(Embedding(max_features,
                                 embedding_dims,
                                 input_length=self.maxlen))

        # we add a GlobalAveragePooling1D, which will average the embeddings
        # of all words in the document
        self.model.add(GlobalAveragePooling1D())

        # We project onto a single unit output layer, and squash via sigmoid:
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])
        self.model.fit(x_train, Y, batch_size=batch_size, epochs=epochs, verbose=2)

def get_most_common_embeddings(tokenizer, nlp):
    import operator
    most_common = list(map(operator.itemgetter(0), sorted(tokenizer.word_index.items(), key=operator.itemgetter(1))))
    n = len(tokenizer.word_index)
    if tokenizer.num_words is not None:
        most_common = most_common[:tokenizer.num_words]
        n = min(tokenizer.num_words, n)
    embeddings = np.zeros((n + 1, nlp.vocab[0].vector.shape[0]), dtype='float32')
    tokenized = nlp.tokenizer.pipe([x for x in most_common])
    for i, lex in enumerate(tokenized):
        if lex.has_vector:
            embeddings[i + 1] = lex.vector
    return embeddings

class CNNClassifier:
    def __init__(self, nlp):
        self.nlp = nlp
        pass
    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)
    def predict_proba(self, X):
        x_test = self.tokenizer.texts_to_sequences(X)
        x_test = sequence.pad_sequences(x_test, maxlen=self.maxlen)
        a = self.model.predict(x_test, verbose=0).flatten()
        a = a.reshape(-1, 1)
        return np.hstack((1 - a, a))
    def fit(self, X, Y, max_features=20000, maxlen=400,
            batch_size=32, hidden_dims=250, filters=250, kernel_size=3,
            epochs=5):
        from keras.preprocessing import sequence
        from keras.models import Sequential
        from keras.layers import Dense, Dropout, Activation
        from keras.layers import Embedding
        from keras.layers import Conv1D, GlobalMaxPooling1D
        self.tokenizer = keras.preprocessing.text.Tokenizer(
            num_words=max_features, split=" ", char_level=False)
        self.tokenizer.fit_on_texts(X)
        x_train = self.tokenizer.texts_to_sequences(X)
        self.maxlen = maxlen
        embeddings = get_most_common_embeddings(self.tokenizer, self.nlp)
        x_train = sequence.pad_sequences(x_train, maxlen=self.maxlen)
        self.model = Sequential()
        # we start off with an efficient embedding layer which maps
        # our vocab indices into embedding_dims dimensions
        self.model.add(
            Embedding(
                embeddings.shape[0],
                embeddings.shape[1],
                input_length=maxlen,
                trainable=False,
                weights=[embeddings]
            )
        )

        self.model.add(Dropout(0.2))

        # we add a Convolution1D, which will learn filters
        # word group filters of size filter_length:
        self.model.add(Conv1D(filters, kernel_size, padding='valid',
                              activation='relu', strides=1))
        # we use max pooling:
        self.model.add(GlobalMaxPooling1D())

        # We add a vanilla hidden layer:
        self.model.add(Dense(hidden_dims))
        self.model.add(Dropout(0.2))
        self.model.add(Activation('relu'))
        # We project onto a single unit output layer, and squash it with a sigmoid:
        self.model.add(Dense(1))
        # model.add(Dense(3))
        self.model.add(Activation('sigmoid'))



        # optimizer = keras.optimizers.Adam(lr=0.001)
        optimizer = keras.optimizers.Adam(lr=0.0001)
        # model.compile(loss='categorical_crossentropy',
        #               optimizer=optimizer,
        #               metrics=['accuracy'])
        self.model.compile(loss='binary_crossentropy',
                           optimizer=optimizer,
                           metrics=['accuracy'])

        self.model.fit(x_train, Y, batch_size=batch_size, epochs=epochs, verbose=2)
