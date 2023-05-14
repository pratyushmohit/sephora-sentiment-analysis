import re

import contractions
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from sentiment_analysis.preprocessing.baseclass import BasePreprocessor


class Preprocessor(BasePreprocessor):
    def __init__(self, path):
        self.path = path
        self.dataset = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def ingest(self):
        list_of_dataframes = []

        if len(self.path) == 1:
            self.dataset = pd.read_csv(self.path)
        elif len(self.path) > 1:
            for path in self.path:
                self.dataset = pd.read_csv(path)
                list_of_dataframes.append(self.dataset)
            self.dataset = pd.concat(list_of_dataframes, axis=0)
            self.dataset.reset_index(drop=True, inplace=True)
        else:
            raise FileNotFoundError("Please input a valid dataset path.")
        
        return self.dataset
    
    def split_data(self):
        #only working with these features for now
        self.dataset = self.dataset.loc[:, ["rating", "review_text", "review_title", "brand_name", "price_usd"]]
        self.dataset.dropna(inplace=True)
        y = self.dataset["rating"]
        x = self.dataset.drop("rating", axis=1)

        if x.shape[0] == y.shape[0]:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, stratify=y, test_size=0.1)

        return self.x_train, self.x_test, self.y_train, self.y_test
    
    def preprocess_text(self, input_text):
        input_text = contractions.fix(input_text)
        input_text = re.sub('\W+',' ', input_text)
        preprocessed_text = input_text.lower()
        return preprocessed_text
    
    def preprocess_text_batch(self, batch, feature):
        batch_output = []
        batch = batch[feature]
        
        for item in tqdm(batch):
            output = self.preprocess_text(item)
            batch_output.append(output)

        return batch_output
    
    def preprocess_categorical(self, batch, feature):
        batch_output = []
        batch = batch[feature]

        for item in tqdm(batch):
            item = re.sub('\W+',' ', item)
            item = item.lower()  
            batch_output.append(item)

        return batch_output

    def preprocess_numerical(self, batch, feature):
        batch = batch[feature]
        return batch
    
    def preprocess_y(self, y):
        y = to_categorical(y)
        return y
    
    def tokenization(self, batch_output):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(batch_output)
        vocabulary = tokenizer.word_index
        size_of_vocabulary = len(vocabulary) + 1
        return tokenizer, vocabulary, size_of_vocabulary
    
    def glove_embedding(self, vocabulary, size_of_vocabulary, dimensions):
        glove_embeddings_index = {}

        f = open(r'sentiment_analysis\\preprocessing\\glove.840B.300d.txt', encoding='utf8')
        for line in f:
            values = line.split()
            word = ''.join(values[:-dimensions])
            coeffs = np.asarray(values[-dimensions:], dtype='float32')
            glove_embeddings_index[word] = coeffs
        f.close()

        embedding_matrix = np.zeros((size_of_vocabulary, dimensions))

        for word, word_index in vocabulary.items():
            embedding_vector = glove_embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[word_index] = embedding_vector
        return embedding_matrix
    
    def padding(self, sequences, max_len):
        padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
        return padded_sequences
    
    def save_data(self, dataframe, filename):
        path = f"sentiment_analysis\data\preprocessed_data\{filename}"

        if isinstance(dataframe, np.ndarray):
            np.save(path, dataframe, allow_pickle=True, fix_imports=True)
        else:
            dataframe.to_csv(path)

        return f"Data saved to {path}"