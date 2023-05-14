import json
import os
import pandas as pd
from . import app
from fastapi import Response
from sentiment_analysis.data_model import PreprocessItem
from sentiment_analysis.preprocessing.preprocessing import Preprocessor

@app.get('/status')
def status():
    json_str = json.dumps({"status": "Running"}, indent=4, default=str)
    return Response(content=json_str, media_type='application/json')

@app.get('/')
async def index():
    return {"message": "Sephora Sentiment Analysis"}

@app.post('/preprocess')
async def preprocess(request: PreprocessItem):
    filenames = request.dict()
    filenames = filenames["filenames"]
    output_folder = "sentiment_analysis\data\preprocessed_data"

    if not os.listdir(output_folder):
        preprocessor = Preprocessor(filenames)
        output = preprocessor.ingest()
        x_train, x_test, y_train, y_test = preprocessor.split_data()

        review_text = preprocessor.preprocess_text_batch(x_train["review_text"])
        review_title = preprocessor.preprocess_text_batch(x_train["review_title"])
        brand_name = preprocessor.preprocess_categorical(x_train["brand_name"])
        price_usd = preprocessor.preprocess_numerical(x_train["price_usd"])


        dataset = pd.DataFrame({"review_text": review_text,
                                "review_title": review_title,
                                "brane_name": brand_name,
                                "price_usd": price_usd})
        print(dataset)
        output = preprocessor.save_data(output)
    else:
        output = "Preprocessing has already been performed."

    return output