import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.algorithms.svd import SVDRecommender
from src.algorithms.item_cf import ItemCFRecommender
from src.algorithms.popularity import PopularityRecommender
from src.algorithms.als import ALSRecommender
from src.evaluation.metrics import get_top_n_recommendations, evaluate, rmse

print("Script started")

def load_splits(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def main():
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

    print("\n--- Results ---")
    print(f"{'Metric':<15} {'SVD':>10} {'Item-CF':>10} {'ALS':>10} {'Popularity':>12}")
    print("-" * 60)
    for metric in svd_results:
        print(f"{metric:<15} {svd_results[metric]:>10.4f} {cf_results[metric]:>10.4f} {als_results[metric]:>10.4f} {pop_results[metric]:>12.4f}")
    print(f"\n{'RMSE':<15} {svd_rmse:>10.4f} {'N/A':>10} {'N/A':>10} {'N/A':>12}")


if __name__ == "__main__":
    main()