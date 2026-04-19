import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.algorithms.als import ALSRecommender
from src.evaluation.metrics import evaluate

print("Loading splits...")
train = pd.read_csv('data/splits/train_ratings.csv')
test = pd.read_csv('data/splits/test_ratings.csv')
print(f"Train: {len(train)} ratings")

print("\nTraining ALS...")
model = ALSRecommender(factors=100, iterations=50, alpha=20)
model.train(train)

print("\nTesting recommend_all on 100 users...")
test_small = test[test['userId'].isin(test['userId'].unique()[:100])]
recs = model.recommend_all(test, train, n=10, sample_users=1000)
print(f"Generated recommendations for {len(recs)} users")
# Check average recommendation list length
avg_len = sum(len(v) for v in recs.values()) / len(recs)
print(f"Average recommendations per user: {avg_len:.2f}")

print("\nEvaluating...")
test_sampled = test[test['userId'].isin(recs.keys())]
results = evaluate(recs, train, test_sampled, n=10, threshold=4.0)
for metric, score in results.items():
    print(f"{metric}: {score:.4f}")