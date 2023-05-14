import numpy as np
import tensorflow as tf
from keras.layers import (LSTM, Concatenate, Dense, Dropout, Embedding,
                          Flatten, Input)
from keras.models import Model
from keras.optimizers import Nadam


class SentimentAnalysisModel():
    def __init__(self, embeddings, sequences, numerical_feature, y_train, data_config, model_config):
        self.embeddings = embeddings
        self.sequences = sequences
        self.numerical_feature = numerical_feature
        self.y_train = y_train
        self.data_config = data_config
        self.model_config = model_config

    def load_data(self):
        self.embedding_matrix_review_text = np.load(self.embeddings[0])
        self.embedding_matrix_review_title = np.load(self.embeddings[1])
        self.padded_sequences_review_text = np.load(self.sequences[0])
        self.padded_sequences_review_title = np.load(self.sequences[1])
        self.sequences_brand_name = np.load(self.sequences[3])
        self.price_usd = np.load(self.numerical_feature)
        self.y_train = np.load(self.y_train)

    def model(self):
        self.load_data()
        tf.keras.backend.clear_session()

        input_layer_1 = Input(
            shape=(self.data_config["review_text"]["max_len"],))
        embedding_1 = Embedding(input_dim=self.data_config["review_text"]["size_of_vocab"],
                                output_dim=300,
                                weights=[self.embedding_matrix_review_text],
                                input_length=self.data_config["review_text"]["max_len"],
                                trainable=False)(input_layer_1)
        lstm = LSTM(100, return_sequences=True)(embedding_1)
        flatten_1 = Flatten()(lstm)

        input_layer_2 = Input(
            shape=(self.data_config["review_title"]["max_len"],))
        embedding_2 = Embedding(input_dim=self.data_config["review_title"]["size_of_vocab"],
                                output_dim=50,
                                weights=[self.embedding_matrix_review_title],
                                input_length=self.data_config["review_title"]["max_len"],
                                trainable=True)(input_layer_2)
        flatten_2 = Flatten()(embedding_2)

        input_layer_3 = Input(shape=(self.data_config["brand_name"]["max_len"],))
        embedding_3 = Embedding(input_dim=self.data_config["brand_name"]["size_of_vocab"],
                                output_dim=2, input_length=1, trainable=True)(input_layer_3)
        flatten_3 = Flatten()(embedding_3)

        input_layer_4 = Input(shape=(1,))
        dense = Dense(50, activation='relu')(input_layer_4)

        concat = Concatenate()([flatten_1, flatten_2, flatten_3, dense])

        dense_after_concat_1 = Dense(100, activation='relu')(concat)
        dropout_1 = Dropout(0.5)(dense_after_concat_1)
        dense_after_concat_2 = Dense(50, activation='relu')(dropout_1)
        dropout_2 = Dropout(0.5)(dense_after_concat_2)
        dense_after_concat_3 = Dense(10, activation='relu')(dropout_2)
        output = Dense(5, activation='softmax')(dense_after_concat_3)

        model = Model(inputs=[input_layer_1, input_layer_2, input_layer_3, input_layer_4], outputs=[output])

        optimizer = Nadam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)

        all_inputs_train = [self.padded_sequences_review_text,
                            self.padded_sequences_review_title,
                            self.sequences_brand_name,
                            self.price_usd]
        
        all_inputs_test = []

        history = model.fit(x=all_inputs_train, 
                            y=self.class_labels, 
                            epochs=10, 
                            batch_size=256, 
                            validation_data=(all_inputs_test, y_test), 
                            callbacks=callbacks)

        print(model.summary())