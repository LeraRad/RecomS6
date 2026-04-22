import sys
from dotenv import load_dotenv
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.algorithms.svd import SVDRecommender
from src.algorithms.item_cf import ItemCFRecommender
from src.algorithms.popularity import PopularityRecommender
from src.algorithms.als import ALSRecommender
from src.evaluation.metrics import get_top_n_recommendations, evaluate, rmse
from src.algorithms.lightfm_recommender import LightFMRecommender
from src.data.feature_engineering import build_movie_features
from src.algorithms.graph_recommender import GraphRecommender

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

print("Script started")

def main():
    def load_splits(train_path, test_path):
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        return train, test

    # --- Movie Features ---
    print("\nBuilding movie features...")
    eligible_movies = set(train['movieId'].unique())
    item_features, movie_index, tag_index = build_movie_features(
        'data/raw/genome_scores.csv',
        'data/raw/genome_tags.csv',
        eligible_movies,
        relevance_threshold=0.6
    )

    train_path = os.path.join('data', 'splits', 'train_ratings.csv')
    test_path = os.path.join('data', 'splits', 'test_ratings.csv')

    print("Loading splits...")
    train, test = load_splits(train_path, test_path)
    print(f"Train: {len(train)} ratings | Test: {len(test)} ratings")

    # --- SVD ---
    print("\nTraining SVD...")
    svd = SVDRecommender(n_factors=50, n_epochs=20, biased=True)
    svd.train(train)
    print("Training complete.")

    print("Generating SVD recommendations...")
    svd_recs = get_top_n_recommendations(svd, test, train, n=10)

    print("Evaluating SVD...")
    svd_results = evaluate(svd_recs, train, test, n=10, threshold=4.0)

    print("Computing SVD RMSE...")
    svd_rmse = rmse(svd, test)

    # --- Item-CF ---
    print("\nTraining Item-CF...")
    item_cf = ItemCFRecommender(min_ratings=50, n_similar=20)
    item_cf.train(train)

    print("Generating Item-CF recommendations (10K sample)...")
    cf_recs = item_cf.recommend_all(test, train, n=10, sample_users=10000)

    print("Evaluating Item-CF...")
    test_sampled = test[test['userId'].isin(cf_recs.keys())]
    cf_results = evaluate(cf_recs, train, test_sampled, n=10, threshold=4.0)

    # --- Popularity ---
    print("\nTraining Popularity baseline...")
    popularity = PopularityRecommender()
    popularity.train(train)
    pop_recs = popularity.recommend_all(test, train, n=10)

    print("Evaluating Popularity...")
    pop_results = evaluate(pop_recs, train, test, n=10, threshold=4.0)

    # --- ALS ---
    print("\nTraining ALS...")
    als = ALSRecommender(factors=50, iterations=20, alpha=40)
    als.train(train)

    print("Generating ALS recommendations (10K sample)...")
    als_recs = als.recommend_all(test, train, n=10, sample_users=10000)

    print("Evaluating ALS...")
    test_sampled_als = test[test['userId'].isin(als_recs.keys())]
    als_results = evaluate(als_recs, train, test_sampled_als, n=10, threshold=4.0)

    # --- LightFM ---
    print("\nTraining LightFM...")
    lightfm = LightFMRecommender(no_components=50, loss='warp', epochs=30)
    lightfm.train(train, item_features_matrix=item_features, movie_index=movie_index)

    print("Generating LightFM recommendations (10K sample)...")
    lfm_recs = lightfm.recommend_all(test, train, n=10, sample_users=10000)

    print("Evaluating LightFM...")
    test_sampled_lfm = test[test['userId'].isin(lfm_recs.keys())]
    lfm_results = evaluate(lfm_recs, train, test_sampled_lfm, n=10, threshold=4.0)

    # --- Graph ---
    print("\nInitializing Graph Recommender...")
    graph = GraphRecommender(
        uri=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASSWORD
    )
    graph.train(train)

    print("Generating Graph recommendations (200 users from graph subset)...")
    graph_users = test[test['userId'].isin(graph.user_ids_in_graph)]['userId'].unique()
    graph_recs = graph.recommend_all(
        test[test['userId'].isin(graph_users)],
        train, n=10, sample_users=200
    )

    print("Evaluating Graph...")
    test_sampled_graph = test[test['userId'].isin(graph_recs.keys())]
    graph_results = evaluate(graph_recs, train, test_sampled_graph, n=10, threshold=4.0)
    graph.close()

    print("\n--- Results ---")
    print(f"{'Metric':<15} {'SVD':>8} {'Item-CF':>8} {'ALS':>8} {'LightFM':>8} {'Graph':>8} {'Popularity':>10}")
    print("-" * 70)
    for metric in svd_results:
        print(f"{metric:<15} {svd_results[metric]:>8.4f} {cf_results[metric]:>8.4f} {als_results[metric]:>8.4f} {lfm_results[metric]:>8.4f} {graph_results[metric]:>8.4f} {pop_results[metric]:>10.4f}")
    print(f"\n{'RMSE':<15} {svd_rmse:>8.4f} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>10}")


if __name__ == "__main__":
    main()