# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 17:33:02 2022

@author: Meghana
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
import json
import bentoml
import numpy as np
from joblib import dump, load

userToMovie="movie_10000.json"
with open(userToMovie, 'r') as dataset:
    data= json.load(dataset)

df=pd.DataFrame(data)
df1=df.T
df1.drop(columns=["id","tmdb_id","imdb_id","title","original_title","adult","belongs_to_collection","homepage","overview","poster_path","production_countries","spoken_languages","message","original_language","production_companies","release_date","status","genres"],inplace=True)
df1=df1.astype(float)
#print(df1["popularity"].describe())
target=df1["popularity"]
features=df1.drop(columns=["popularity"])
features["budget"].fillna(features["budget"].mean(),inplace=True)
features["revenue"].fillna(features["revenue"].mean(),inplace=True)
features["runtime"].fillna(features["runtime"].mean(),inplace=True)
features["vote_average"].fillna(features["vote_average"].mean(),inplace=True)
features["vote_count"].fillna(features["vote_count"].mean(),inplace=True)
target.fillna(target.mean(),inplace=True)
x=features.values
y=target.values
lr=LinearRegression()
lr.fit(x,y)
#print(lr.predict([[10000000,500000,154,6.4,1000]]))
dump(lr, "movie_popularity.joblib")
#bentoml.sklearn.save("movie_popularity", lr)
#runner = bentoml.sklearn.load_runner("movie_popularity:latest")
#print(runner.run(np.array([10000000,500000,154,6.4,1000])))
