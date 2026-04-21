import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from lightfm import LightFM
from lightfm.data import Dataset


class LightFMRecommender:
    """
    Hybrid recommender using LightFM.
    Combines collaborative filtering with movie genome tag features.
    Uses WARP loss — optimized directly for Top-N ranking.
    """

    def __init__(self, no_components=50, loss='warp', epochs=30, num_threads=4):
        self.no_components = no_components
        self.loss = loss
        self.epochs = epochs
        self.num_threads = num_threads
        self.model = None
        self.dataset = None
        self.user_id_map = None
        self.item_id_map = None
        self.user_index = None
        self.movie_index = None
        self.index_to_movie = None
        self.interactions = None
        self.item_features = None

    def train(self, train_ratings: pd.DataFrame, item_features_matrix=None,
              movie_index=None):
        """
        Train LightFM model.

        Args:
            train_ratings: DataFrame with columns [userId, movieId, rating]
            item_features_matrix: sparse matrix (n_movies x n_tags) from feature engineering
            movie_index: dict mapping movieId to feature matrix row index
        """
        print("Building LightFM dataset...")
        self.dataset = Dataset()

        # Fit dataset with all users and items
        self.dataset.fit(
            users=train_ratings['userId'].unique(),
            items=train_ratings['movieId'].unique()
        )

        # Store mappings
        self.user_id_map, _, self.item_id_map, _ = self.dataset.mapping()
        self.movie_index = movie_index
        self.index_to_movie = {idx: mid for mid, idx in self.item_id_map.items()}

        # Build interaction matrix
        # Use ratings >= 3.5 as positive interactions
        # LightFM WARP works best with positive-only interactions
        print("Building interaction matrix...")
        positive = train_ratings[train_ratings['rating'] >= 3.5]
        (self.interactions, _) = self.dataset.build_interactions(
            [(row['userId'], row['movieId'])
             for _, row in positive.iterrows()]
        )
        print(f"Interactions shape: {self.interactions.shape}")
        print(f"Positive interactions: {self.interactions.nnz}")

        # Build item features if provided
        self.item_features = None
        if item_features_matrix is not None and movie_index is not None:
            print("Aligning item features with LightFM item index...")
            n_items = len(self.item_id_map)
            n_tags = item_features_matrix.shape[1]

            # Reorder feature matrix rows to match LightFM's internal item ordering
            aligned = np.zeros((n_items, n_tags))
            for movie_id, lightfm_idx in self.item_id_map.items():
                if movie_id in movie_index:
                    feature_row = movie_index[movie_id]
                    aligned[lightfm_idx] = item_features_matrix[feature_row].toarray()

            self.item_features = csr_matrix(aligned)
            print(f"Item features shape: {self.item_features.shape}")

        # Train model
        print(f"Training LightFM with {self.loss} loss...")
        self.model = LightFM(
            no_components=self.no_components,
            loss=self.loss
        )
        self.model.fit(
            self.interactions,
            item_features=self.item_features,
            epochs=self.epochs,
            num_threads=self.num_threads,
            verbose=True
        )
        print("Training complete.")

    def recommend_all(self, test_ratings, train_ratings, n=10, sample_users=None):
        """
        Generate top-N recommendations for all users.
        """
        test_users = test_ratings['userId'].unique()

        if sample_users and sample_users < len(test_users):
            np.random.seed(42)
            test_users = np.random.choice(test_users, size=sample_users, replace=False)
            print(f"Sampled {sample_users} users from {len(test_users)} total")

        # Build seen movies per user
        user_train_movies = (
            train_ratings.groupby('userId')['movieId']
            .apply(set)
            .to_dict()
        )

        n_items = len(self.item_id_map)
        all_item_indices = np.arange(n_items)

        print("Generating recommendations...")
        recommendations = {}
        for i, user_id in enumerate(test_users):
            if i % 10000 == 0:
                print(f"  {i}/{len(test_users)} users processed...")

            if user_id not in self.user_id_map:
                continue

            user_idx = self.user_id_map[user_id]

            # Score all items
            scores = self.model.predict(
                user_idx,
                all_item_indices,
                item_features=self.item_features
            )

            # Mask seen items
            seen = user_train_movies.get(user_id, set())
            seen_indices = [
                self.item_id_map[m] for m in seen
                if m in self.item_id_map
            ]
            scores[seen_indices] = -np.inf

            # Get top-N
            top_indices = np.argpartition(scores, -n)[-n:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
            recommendations[user_id] = [
                self.index_to_movie[idx] for idx in top_indices
            ]

        return recommendations