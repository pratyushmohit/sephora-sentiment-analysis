import tensorflow as tf
from keras.layers import (LSTM, Concatenate, Dense, Dropout, Embedding,
                          Flatten, Input)
from keras.models import Model
from keras.optimizers import Nadam


class SentimentAnalysisModel():
    def __init__(self, data_config, model_config):
        self.data_config = data_config
        self.model_config = model_config

    def model(self):
        tf.keras.backend.clear_session()

        input_layer_1 = Input(
            shape=(self.data_config["review_text"]["max_len"],))
        embedding_1 = Embedding(input_dim=self.data_config["review_text"]["size_of_vocab"], output_dim=300, weights=[embedding_matrix_review_text], input_length=max_len_review_text,
                                trainable=False)(input_layer_1)
        lstm = LSTM(100, return_sequences=True)(embedding_1)
        flatten_1 = Flatten()(lstm)

        input_layer_2 = Input(
            shape=(self.data_config["review_title"]["max_len"],))
        embedding_2 = Embedding(input_dim=self.data_config["review_title"]["size_of_vocab"], output_dim=50, weights=[embedding_matrix_review_title], input_length=max_len_review_title,
                                trainable=True)(input_layer_2)
        flatten_2 = Flatten()(embedding_2)

        input_layer_3 = Input(shape=(3,))
        embedding_3 = Embedding(input_dim=project_grade_category_size_of_vocabulary,
                                output_dim=2, input_length=1, trainable=True)(input_layer_3)
        flatten_3 = Flatten()(embedding_3)

        input_layer_4 = Input(shape=(clean_categories_max_len,))
        embedding_4 = Embedding(input_dim=clean_categories_size_of_vocabulary, output_dim=2, input_length=clean_categories_max_len,
                                trainable=True)(input_layer_4)
        flatten_4 = Flatten()(embedding_4)

        input_layer_5 = Input(shape=(clean_subcategories_max_len,))
        embedding_5 = Embedding(input_dim=clean_subcategories_size_of_vocabulary, output_dim=2, input_length=clean_subcategories_max_len,
                                trainable=True)(input_layer_5)
        flatten_5 = Flatten()(embedding_5)

        input_layer_6 = Input(shape=(1,))
        embedding_6 = Embedding(input_dim=teacher_prefix_size_of_vocabulary,
                                output_dim=2, input_length=1, trainable=True)(input_layer_6)
        flatten_6 = Flatten()(embedding_6)

        input_layer_7 = Input(shape=(2,))
        dense = Dense(50, activation='relu')(input_layer_7)
