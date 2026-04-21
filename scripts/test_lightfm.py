import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.algorithms.lightfm_recommender import LightFMRecommender
from src.data.feature_engineering import build_movie_features
from src.evaluation.metrics import evaluate

print("Loading splits...")
train = pd.read_csv('data/splits/train_ratings.csv')
test = pd.read_csv('data/splits/test_ratings.csv')
print(f"Train: {len(train)} ratings")

print("\nBuilding movie features...")
eligible_movies = set(train['movieId'].unique())
item_features, movie_index, tag_index = build_movie_features(
    'data/raw/genome_scores.csv',
    'data/raw/genome_tags.csv',
    eligible_movies,
    relevance_threshold=0.6
)

print("\nTraining LightFM...")
model = LightFMRecommender(no_components=50, loss='warp', epochs=30)
model.train(train, item_features_matrix=item_features, movie_index=movie_index)

print("\nTesting recommend_all on 1000 users...")
recs = model.recommend_all(test, train, n=10, sample_users=1000)
print(f"Generated recommendations for {len(recs)} users")
avg_len = sum(len(v) for v in recs.values()) / len(recs)
print(f"Average recommendations per user: {avg_len:.2f}")
print("Done.")

print("\nEvaluating...")
test_sampled = test[test['userId'].isin(recs.keys())]
results = evaluate(recs, train, test_sampled, n=10, threshold=4.0)
for metric, score in results.items():
    print(f"{metric}: {score:.4f}")