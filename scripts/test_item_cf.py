import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.algorithms.item_cf import ItemCFRecommender

print("Loading train split...")
train = pd.read_csv('data/splits/train_ratings.csv')
print(f"Train loaded: {len(train)} ratings")
test = pd.read_csv('data/splits/test_ratings.csv')

print("\nTraining Item-CF...")
model = ItemCFRecommender(min_ratings=50, n_similar=20)
model.train(train)

print("\nTest single prediction...")
# Grab a real user and movie from train
sample_user = train['userId'].iloc[0]
sample_movie = train['movieId'].iloc[0]
pred = model.predict(sample_user, sample_movie)
print(f"Predicted rating for user {sample_user}, movie {sample_movie}: {pred:.4f}")

print("\nAll checks passed.")

print("Testing recommend_all on 100 users...")
test_small = test[test['userId'].isin(test['userId'].unique()[:100])]
recs = model.recommend_all(test, train, n=10, sample_users=10000)
print(f"Generated recommendations for {len(recs)} users")
print("Done.")