#!/bin/sh

echo "\nStarting Sephora Sentiment Analysis..."

python -m uvicorn sentiment_analysis.main:app --reload
