import csv
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("final1.csv")
df = df[df["soup"].notna()]
count = CountVectorizer(stop_words = "english")
countMatrics = count.fit_transform(df["soup"])
cosine_sim = cosine_similarity(countMatrics, countMatrics)
df = df.reset_index()

indices = pd.Series(df.index, index = df["original_title"])

def getRecommendation(title):
    idx = indices[title]
    simScores = list(enumerate(cosine_sim[idx]))
    simScores = sorted(simScores, key = lambda x: x[1], reverse = True)
    simScores = simScores[1:11] 
    movieIndices = [i[0] for i in simScores]
    return df[['content_id', "total_votes"]].loc[movieIndices].values.tolist()
