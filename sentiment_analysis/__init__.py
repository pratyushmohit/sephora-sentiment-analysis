from fastapi import FastAPI

app = FastAPI(title="Sephora Sentiment Analysis",
              description="A sentiment analysis pipeline for Sephora Product and Skincare Reviews dataset.",
              version="0.1.0")