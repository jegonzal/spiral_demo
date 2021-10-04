import nltk
import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class Function:
    def __init__(self):
        nltk.download('vader_lexicon')
        sid = SentimentIntensityAnalyzer()
        self.model = lambda sentence: sid.polarity_scores(sentence)["compound"]
        

    def predict(self, df):
        predictions = df["REVIEW"].apply(self.model)
        df['POSITIVITY'] = predictions
        print(df.columns)
        return df
