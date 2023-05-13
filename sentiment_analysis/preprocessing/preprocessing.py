from baseclass import BasePreprocessor
import pandas as pd
from typing import Coroutine, Any
from sklearn.model_selection import train_test_split
import re
import contractions
from tqdm import tqdm


class Preprocessor(BasePreprocessor):
    def __init__(self, path):
        self.path = path
        self.dataset = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    async def ingest(self):
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
    
    async def split_data(self):
        #only working with these features for now
        self.dataset = self.dataset.loc[:, ["rating", "review_text", "review_title", "brand_name", "price_usd"]]
        self.dataset.dropna(inplace=True)
        y = self.dataset["rating"]
        x = self.dataset.drop("rating", axis=1)

        if x.shape[0] == y.shape[0]:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, stratify=y, test_size=0.1)

        return self.x_train, self.x_test, self.y_train, self.y_test
    
    async def preprocess(self, input_text):
        input_text = contractions.fix(input_text)
        input_text = re.sub('\W+',' ', input_text)
        preprocessed_text = input_text.lower()
        return preprocessed_text
    
    async def preprocess_batch(self, batch):
        batch_output = []
        
        for item in tqdm(batch):
            output = self.preprocess(item)
            batch_output.append(output)
        return batch_output