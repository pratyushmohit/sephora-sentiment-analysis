import json
from . import app
from fastapi import Response

@app.get('/status')
def status():
    json_str = json.dumps({"status": "Running"}, indent=4, default=str)
    return Response(content=json_str, media_type='application/json')

@app.get('/')
async def index():
    return {"message": "Sephora Sentiment Analysis"}