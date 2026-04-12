from src.algorithms.popularity import PopularityRecommender
from src.evaluation.metrics import get_top_n_recommendations, evaluate, rmse
from src.algorithms.svd import SVDRecommender
import sys
import os
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

    # SVD
    print("\nTraining SVD...")
    svd = SVDRecommender(n_factors=50, n_epochs=20, biased=True)
    svd.train(train)
    print("Training complete.")

    print("Generating SVD recommendations...")
    svd_recs = get_top_n_recommendations(svd, test, train, n=10)

    print("Evaluating SVD...")
    svd_results = evaluate(svd_recs, train, test, n=10, threshold=4.0)

    print("\nComputing RMSE...")
    svd_rmse = rmse(svd, test)
    print(f"SVD RMSE: {svd_rmse:.4f}")

    # Popularity baseline
    print("\nTraining Popularity baseline...")
    popularity = PopularityRecommender()
    popularity.train(train)
    pop_recs = popularity.recommend_all(test, train, n=10)

    print("Evaluating Popularity baseline...")
    pop_results = evaluate(pop_recs, train, test, n=10, threshold=4.0)

    # Results
    print("\n--- Results ---")
    print(f"{'Metric':<15} {'SVD':>10} {'Popularity':>12}")
    print("-" * 40)
    for metric in svd_results:
        print(f"{metric:<15} {svd_results[metric]:>10.4f} {pop_results[metric]:>12.4f}")


if __name__ == "__main__":
    main()