import json

from fastapi import Response

from sentiment_analysis.data_model import PreprocessItem
from sentiment_analysis.pipeline.pipeline import Pipeline

from . import app


@app.get('/status')
def status():
    output = json.dumps({"status": "Running"}, indent=4, default=str)
    return Response(content=output, media_type='application/json')

@app.get('/')
async def index():
    return {"message": "Sephora Sentiment Analysis"}

@app.post('/preprocess')
async def preprocess(request: PreprocessItem):
    filenames = request.dict()
    filenames = filenames["filenames"]
    pipeline = Pipeline(filenames)
    output = pipeline.preprocessing_pipeline()
    return Response(content=output, media_type='application/json')