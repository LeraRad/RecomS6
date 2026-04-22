import sys
from dotenv import load_dotenv
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.algorithms.graph_recommender import GraphRecommender
from src.evaluation.metrics import evaluate

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

print("Loading splits...")
train = pd.read_csv('data/splits/train_ratings.csv')
test = pd.read_csv('data/splits/test_ratings.csv')

print("\nInitializing Graph Recommender...")
graph = GraphRecommender(
        uri=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASSWORD
    )
graph.train(train)

print("\nTesting recommend_all on 200 users...")
graph_users = test[test['userId'].isin(graph.user_ids_in_graph)]['userId'].unique()
print(f"Test users in graph: {len(graph_users)}")

recs = graph.recommend_all(
    test[test['userId'].isin(graph_users)],
    train, n=10, sample_users=200
)
print(f"Generated recommendations for {len(recs)} users")
avg_len = sum(len(v) for v in recs.values()) / len(recs) if recs else 0
print(f"Average recommendations per user: {avg_len:.2f}")

print("\nEvaluating...")
test_sampled = test[test['userId'].isin(recs.keys())]
results = evaluate(recs, train, test_sampled, n=10, threshold=4.0)
for metric, score in results.items():
    print(f"{metric}: {score:.4f}")

graph.close()
print("Done.")