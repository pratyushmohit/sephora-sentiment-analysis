# Sentiment Analysis on Sephora Product and Skincare Reviews

![Sephora Sentiment Analyis](https://github.com/pratyushmohit/sephora-sentiment-analysis/blob/main/blob/cover.jpg)

# Steps to execute the pipeline

1. Download Sephora Products and Skincare Reviews dataset [here](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews). 

2. Download GloVe (Global Vectors for Word Representation) [here](https://nlp.stanford.edu/projects/glove/). Please download the 840B version of GloVe and copy it into the preprocessing folder. If you would like to download a smaller version of GloVe, please do so but update the glove_embedding() method in the Preprocessor class with the correct file.