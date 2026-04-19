import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import implicit


class ALSRecommender:
    """
    Alternating Least Squares recommender using implicit feedback.
    Treats explicit ratings as confidence-weighted positive interactions.
    No threshold filtering — confidence weighting handles sentiment.
    """

    def __init__(self, factors=100, iterations=30, alpha=20, regularization=0.1):
        self.factors = factors
        self.iterations = iterations
        self.alpha = alpha
        self.regularization = regularization
        self.model = None
        self.user_index = None
        self.movie_index = None
        self.index_to_movie = None
        self.user_item_sparse = None

    def train(self, train_ratings: pd.DataFrame):
        """
        Train ALS model on ratings data.
        Converts explicit ratings to confidence matrix.

        Args:
            train_ratings: DataFrame with columns [userId, movieId, rating]
        """
        print("Building user and movie indices...")
        user_ids = train_ratings['userId'].unique()
        movie_ids = train_ratings['movieId'].unique()

        self.user_index = {uid: idx for idx, uid in enumerate(user_ids)}
        self.movie_index = {mid: idx for idx, mid in enumerate(movie_ids)}
        self.index_to_movie = {idx: mid for mid, idx in self.movie_index.items()}
        self.index_to_user = {idx: uid for uid, idx in self.user_index.items()}

        print(f"Max user index: {max(self.user_index.values())}")
        print(f"Number of users: {len(self.user_index)}")

        # Convert ratings to confidence scores
        # confidence = 1 + alpha * rating
        print("Converting ratings to confidence matrix...")
        rows = train_ratings['userId'].map(self.user_index).values
        cols = train_ratings['movieId'].map(self.movie_index).values
        confidence = 1 + self.alpha * train_ratings['rating'].values

        # ALS expects item-user matrix (items as rows)
        user_item = csr_matrix(
            (confidence, (rows, cols)),
            shape=(len(user_ids), len(movie_ids))
        )
        self.user_item_sparse = user_item
        item_user = user_item

        print("Training ALS model...")
        self.model = implicit.als.AlternatingLeastSquares(
            factors=self.factors,
            iterations=self.iterations,
            regularization=self.regularization,
            use_gpu=False
        )
        self.model.fit(item_user)
        print("Training complete.")

        print(f"Model user factors shape: {self.model.user_factors.shape}")
        print(f"Model item factors shape: {self.model.item_factors.shape}")

    def recommend_all(self, test_ratings, train_ratings, n=10, sample_users=None):
        """
        Generate top-N recommendations for all users.
        Uses implicit library's built-in batch recommendation.
        """
        test_users = test_ratings['userId'].unique()

        if sample_users and sample_users < len(test_users):
            np.random.seed(42)
            test_users = np.random.choice(test_users, size=sample_users, replace=False)
            print(f"Sampled {sample_users} users from {len(test_users)} total")

        # Map to internal indices
        user_indices = np.array([
            self.user_index[uid]
            for uid in test_users
            if uid in self.user_index
        ])
        valid_users = [
            uid for uid in test_users
            if uid in self.user_index
        ]

        print("Generating recommendations...")
        # implicit's recommend method handles filtering seen items internally
        ids, scores = self.model.recommend(
            user_indices,
            self.user_item_sparse[user_indices],
            N=n,
            filter_already_liked_items=True
        )

        recommendations = {}
        for i, user_id in enumerate(valid_users):
            recommendations[user_id] = [
            self.index_to_movie[int(movie_idx)]
            for movie_idx in ids[i]
            if int(movie_idx) in self.index_to_movie
        ]

        return recommendations