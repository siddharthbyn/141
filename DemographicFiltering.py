import csv
import pandas as pd
import numpy as np
import plotly.express as px

df = pd.read_csv("final1.csv")
c = df["vote_average"].mean()
m = df["vote_count"].quantile(0.9)
Qarticles = df.copy().loc[df["vote_count"]>=m]

def weightedRating(x,m = m,c = c):
  v = x['vote_count']
  r = x['vote_average']
  return (v/(v+m)*r)+(m/(m+v)*c)

Qarticles['score'] = Qarticles.apply(weightedRating,axis=1)
Qarticles = Qarticles.sort_values('score',ascending= False)
Qarticles[['content_id','vote_count','vote_average','score']].head(10)

fig = px.bar((Qarticles.head(10).sort_values('score',ascending = True)),x = 'score',y = 'content_id',orientation = 'h')
fig.show()

output = Qarticles[['content_id', "total_votes", "vote_average", "overview"]].head(20).values.tolist()
