import json
from . import app
from fastapi import Response
from preprocessing.preprocessing import Preprocessor

@app.get('/status')
def status():
    json_str = json.dumps({"status": "Running"}, indent=4, default=str)
    return Response(content=json_str, media_type='application/json')

@app.get('/')
async def index():
    return {"message": "Sephora Sentiment Analysis"}

@app.post('/preprocess')
async def preprocess(request):
    print(request)
    output = []
    
    preprocessor = Preprocessor()
    output = preprocessor.ingest()
    output = preprocessor.split_data()
    output = preprocessor.preprocess_batch()
    output = preprocessor.save_data(output)
    return output