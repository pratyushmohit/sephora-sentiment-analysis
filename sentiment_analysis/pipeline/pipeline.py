import json
import os

import numpy as np
import pandas as pd

from sentiment_analysis.pipeline.baseclass import BasePipeline
from sentiment_analysis.preprocessing.preprocessing import Preprocessor
from sentiment_analysis.model.model import SentimentAnalysisModel


class Pipeline(BasePipeline):
    def __init__(self, input):
        self.input = input
        self.output = None

    def preprocessing_pipeline(self):
        output_folder = "sentiment_analysis\data\preprocessed_data"
        preprocessor = Preprocessor(self.input)

        if not os.listdir(output_folder):
            output = preprocessor.ingest()
            x_train, x_test, y_train, y_test = preprocessor.split_data()

            review_text = preprocessor.preprocess_text_batch(
                x_train, "review_text")
            review_title = preprocessor.preprocess_text_batch(
                x_train, "review_title")
            brand_name = preprocessor.preprocess_categorical(
                x_train, "brand_name")
            price_usd = preprocessor.preprocess_numerical(x_train, "price_usd")
            y = preprocessor.preprocess_y(y_train)

            x = pd.DataFrame({"review_text": review_text,
                              "review_title": review_title,
                              "brand_name": brand_name,
                              "price_usd": price_usd})

            preprocessor.save_data(x, "x_train_preprocessed.csv")
            preprocessor.save_data(y, "y_train.npy")
        else:
            x = pd.read_csv("sentiment_analysis\\data\preprocessed_data\\x_train_preprocessed.csv",
                            index_col="Unnamed: 0")
            x.reset_index(inplace=True, drop=True)

        # To do: Try to generalize the following code to account for any type of feature and for train set or test
        data_config = {"review_text": {},
                       "review_title": {},
                       "brand_name": {}}

        tokenizer_review_text, vocab_review_text, size_of_vocab_review_text = preprocessor.tokenization(
            x["review_text"])
        embedding_matrix_review_text = preprocessor.glove_embedding(
            vocab_review_text, size_of_vocab_review_text, 300)
        max_len_review_text = max([len(datapoint.split())
                                  for datapoint in x["review_text"]])
        sequences_review_text = tokenizer_review_text.texts_to_sequences(
            x["review_text"])
        padded_sequences_review_text = preprocessor.padding(
            sequences_review_text, max_len_review_text)
        data_config["review_text"]["max_len"] = max_len_review_text
        data_config["review_text"]["size_of_vocab"] = size_of_vocab_review_text

        tokenizer_review_title, vocab_review_title, size_of_vocab_review_title = preprocessor.tokenization(
            x["review_title"])
        embedding_matrix_review_title = preprocessor.glove_embedding(
            vocab_review_title, size_of_vocab_review_title, 50)
        max_len_review_title = max([len(datapoint.split())
                                   for datapoint in x["review_title"]])
        sequences_review_title = tokenizer_review_title.texts_to_sequences(
            x["review_title"])
        padded_sequences_review_title = preprocessor.padding(
            sequences_review_title, max_len_review_title)
        data_config["review_title"]["max_len"] = max_len_review_title
        data_config["review_title"]["size_of_vocab"] = size_of_vocab_review_title

        tokenizer_brand_name, vocab_brand_name, size_of_vocab_brand_name = preprocessor.tokenization(
            x["brand_name"])
        max_len_brand_name = max([len(datapoint.split())
                                 for datapoint in x["brand_name"]])
        sequences_brand_name = np.array(
            tokenizer_brand_name.texts_to_sequences(x['brand_name']), dtype=object)
        data_config["brand_name"]["max_len"] = max_len_brand_name
        data_config["brand_name"]["size_of_vocab"] = size_of_vocab_brand_name

        data_config = json.dumps(data_config, indent=4)
        with open("sentiment_analysis\\model\\data_config.json", "w") as output_file:
            output_file.write(data_config)

        price_usd = x["price_usd"]

        preprocessor.save_data(embedding_matrix_review_text,
                               "embedding_matrix_review_text.npy")
        preprocessor.save_data(padded_sequences_review_text,
                               "padded_sequences_review_text.npy")
        preprocessor.save_data(embedding_matrix_review_title,
                               "embedding_matrix_review_title.npy")
        preprocessor.save_data(padded_sequences_review_title,
                               "padded_sequences_review_title.npy")
        preprocessor.save_data(sequences_brand_name,
                               "sequences_brand_name.npy")
        preprocessor.save_data(price_usd, "price_usd.npy")

        output = json.dumps(
            {"status": "Preprocessing Pipeline successfully executed!"}, indent=4, default=str)
        return output

    def model_pipeline(self):
        print(self.input)
        embeddings = self.input["embeddings"]
        sequences = self.input["sequences"]
        numerical_feature = self.input["numerical_feature"]
        class_label = self.input["class_label"]

        with open("sentiment_analysis\model\data_config.json") as json_file:
            data_config = json.load(json_file)
        
        with open("sentiment_analysis\model\model_config.json") as json_file:
            model_config = json.load(json_file)
        
        model = SentimentAnalysisModel(embeddings, sequences, numerical_feature, class_label, data_config, model_config)
        history = model.model()
        return history
