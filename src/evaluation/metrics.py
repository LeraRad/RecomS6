import numpy as np
import pandas as pd


def get_top_n_recommendations(model, test_ratings, train_ratings, n=10):
    """
    Generate top-N movie recommendations for each user.
    Uses matrix multiplication instead of individual predict() calls for efficiency.
    Only recommends movies the user hasn't seen in training.
    """
    trainset = model.model.trainset

    # Extract factor matrices and biases from trained SVD
    pu = model.model.pu  # user factors (n_users x n_factors)
    qi = model.model.qi  # item factors (n_items x n_factors)
    bu = model.model.bu  # user biases
    bi = model.model.bi  # item biases
    global_mean = trainset.global_mean

    # Compute full scores matrix: (n_users x n_movies)
    scores_matrix = pu @ qi.T + bu[:, np.newaxis] + bi[np.newaxis, :] + global_mean

    # Build mappings: Surprise internal index <-> actual userId/movieId
    inner_to_user = {inner: trainset.to_raw_uid(inner) for inner in range(trainset.n_users)}
    inner_to_movie = {inner: trainset.to_raw_iid(inner) for inner in range(trainset.n_items)}
    user_to_inner = {v: k for k, v in inner_to_user.items()}

    # Build per-user set of already seen movies (in train)
    user_train_movies = (
        train_ratings.groupby('userId')['movieId']
        .apply(set)
        .to_dict()
    )

    all_inner_movies = list(inner_to_movie.keys())
    all_movie_ids = [inner_to_movie[i] for i in all_inner_movies]

    recommendations = {}
    for user_id in test_ratings['userId'].unique():
        if user_id not in user_to_inner:
            # User not seen during training, skip
            continue

        inner_uid = user_to_inner[user_id]
        user_scores = scores_matrix[inner_uid]

        seen_movies = user_train_movies.get(user_id, set())

        # Filter seen movies and get top N
        candidates = [
            (all_movie_ids[idx], user_scores[all_inner_movies[idx]])
            for idx in range(len(all_inner_movies))
            if all_movie_ids[idx] not in seen_movies
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        recommendations[user_id] = [movie_id for movie_id, _ in candidates[:n]]

    return recommendations


def get_relevant_items(test_ratings, threshold=4.0):
    """
    Get ground truth relevant items per user from test split.
    """
    relevant = (
        test_ratings[test_ratings['rating'] >= threshold]
        .groupby('userId')['movieId']
        .apply(set)
        .to_dict()
    )
    return relevant


def precision_at_k(recommendations, relevant_items):
    """
    Precision@K: fraction of recommended items that are relevant.
    """
    scores = []
    for user_id, rec_list in recommendations.items():
        if user_id not in relevant_items or not relevant_items[user_id]:
            continue
        relevant = relevant_items[user_id]
        hits = sum(1 for movie in rec_list if movie in relevant)
        scores.append(hits / len(rec_list))
    return np.mean(scores)


def recall_at_k(recommendations, relevant_items):
    """
    Recall@K: fraction of relevant items that were recommended.
    """
    scores = []
    for user_id, rec_list in recommendations.items():
        if user_id not in relevant_items or not relevant_items[user_id]:
            continue
        relevant = relevant_items[user_id]
        hits = sum(1 for movie in rec_list if movie in relevant)
        scores.append(hits / len(relevant))
    return np.mean(scores)


def ndcg_at_k(recommendations, relevant_items):
    """
    NDCG@K: ranking-aware metric, rewards relevant items appearing higher in list.
    """
    scores = []
    for user_id, rec_list in recommendations.items():
        if user_id not in relevant_items or not relevant_items[user_id]:
            continue
        relevant = relevant_items[user_id]

        dcg = sum(
            1 / np.log2(rank + 2)
            for rank, movie in enumerate(rec_list)
            if movie in relevant
        )
        ideal_hits = min(len(relevant), len(rec_list))
        idcg = sum(1 / np.log2(rank + 2) for rank in range(ideal_hits))

        scores.append(dcg / idcg if idcg > 0 else 0)
    return np.mean(scores)

def rmse(model, test_ratings, sample_size=50000):
    """
    RMSE: root mean squared error between predicted and actual ratings.
    Evaluated on a random sample for efficiency.
    Lower is better.
    """
    sample = test_ratings.sample(n=min(sample_size, len(test_ratings)), random_state=42)
    
    predictions = [
        model.model.predict(row['userId'], row['movieId']).est
        for _, row in sample.iterrows()
    ]
    actual = sample['rating'].values
    return np.sqrt(np.mean((np.array(predictions) - actual) ** 2))


def evaluate(recommendations, train_ratings, test_ratings, n=10, threshold=4.0):
    """
    Run full Top-N evaluation. Returns dict of all metrics.
    Accepts pre-computed recommendations dict {user_id: [movie_id, ...]}
    """
    # Filter test movies not seen in train
    train_movies = set(train_ratings['movieId'].unique())
    test_ratings = test_ratings[test_ratings['movieId'].isin(train_movies)]

    relevant_items = get_relevant_items(test_ratings, threshold)

    return {
        f'precision@{n}': precision_at_k(recommendations, relevant_items),
        f'recall@{n}': recall_at_k(recommendations, relevant_items),
        f'ndcg@{n}': ndcg_at_k(recommendations, relevant_items),
    }