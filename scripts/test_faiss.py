import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import time
from src.algorithms.svd import SVDRecommender

print("Loading train split...")
train = pd.read_csv('data/splits/train_ratings.csv')

print("Training SVD...")
model = SVDRecommender(n_factors=50, n_epochs=20, biased=True)
model.train(train)

# Pick a sample user
sample_user = int(train['userId'].iloc[0])

print("\nTesting standard recommend()...")
start = time.time()
recs_standard = model.recommend(sample_user, n=10)
standard_time = time.time() - start
print(f"Time: {standard_time:.3f}s")
print(f"Top 3: {recs_standard[:3]}")

print("\nTesting recommend_faiss()...")
start = time.time()
recs_faiss = model.recommend_faiss(sample_user, n=10)
faiss_time = time.time() - start
print(f"Time: {faiss_time:.3f}s")
print(f"Top 3: {recs_faiss[:3]}")

print(f"\nSpeedup: {standard_time/faiss_time:.1f}x faster")