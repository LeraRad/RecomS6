import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize


class ItemCFRecommender:
    """
    Item-based collaborative filtering recommender.
    Uses Pearson correlation to compute item-item similarity.
    Only computes similarity for movies with >= min_ratings ratings.
    """

    def __init__(self, min_ratings=50, n_similar=20):
        self.min_ratings = min_ratings
        self.n_similar = n_similar
        self.item_similarity = None
        self.eligible_movies = None
        self.movie_index = None
        self.user_ratings = None


    def train(self, train_ratings: pd.DataFrame):
        """
        Build item-item similarity matrix using Pearson correlation.
        Uses sparse matrices for memory efficiency.

        Args:
            train_ratings: DataFrame with columns [userId, movieId, rating]
        """
        print("Filtering eligible movies...")
        movie_counts = train_ratings.groupby('movieId').size()
        self.eligible_movies = set(
            movie_counts[movie_counts >= self.min_ratings].index
        )
        print(f"Eligible movies: {len(self.eligible_movies)}")

        # Filter to eligible movies only
        filtered = train_ratings[
            train_ratings['movieId'].isin(self.eligible_movies)
        ].copy()

        # Build user and movie index mappings
        print("Building sparse user-item matrix...")
        user_ids = filtered['userId'].unique()
        movie_ids = filtered['movieId'].unique()

        self.user_index = {uid: idx for idx, uid in enumerate(user_ids)}
        self.movie_index = {mid: idx for idx, mid in enumerate(movie_ids)}
        self.index_to_movie = {idx: mid for mid, idx in self.movie_index.items()}

        # Center ratings per user (Pearson requires mean-centering)
        user_means = filtered.groupby('userId')['rating'].mean()
        filtered['rating_centered'] = (
            filtered['rating'] - filtered['userId'].map(user_means)
        )

        # Store global mean and user means for prediction fallback
        self.global_mean = train_ratings['rating'].mean()
        self.user_means = user_means

        # Store raw ratings for prediction
        self.raw_ratings = filtered.set_index(['userId', 'movieId'])['rating']

        # Build sparse matrix with centered ratings
        rows = filtered['userId'].map(self.user_index).values
        cols = filtered['movieId'].map(self.movie_index).values
        data = filtered['rating_centered'].values

        user_item_sparse = csr_matrix(
            (data, (rows, cols)),
            shape=(len(user_ids), len(movie_ids))
        )

        # Compute item-item Pearson similarity
        # Transpose: items as rows, normalize, then dot product
        print("Computing item-item similarity...")
        item_matrix = user_item_sparse.T  # shape: (n_items, n_users)
        item_matrix_normalized = normalize(item_matrix, norm='l2')
        self.item_similarity = (item_matrix_normalized @ item_matrix_normalized.T).toarray()
        print(f"Similarity matrix shape: {self.item_similarity.shape}")
        print("Training complete.")

    def get_similar_items(self, movie_id, n=None):
        """
        Get top-N most similar movies to a given movie.
        """
        n = n or self.n_similar
        if movie_id not in self.movie_index:
            return []

        idx = self.movie_index[movie_id]
        similarities = self.item_similarity[idx]
        top_indices = np.argsort(similarities)[::-1][1:n+1]  # exclude self
        movies = list(self.movie_index.keys())
        return [(movies[i], similarities[i]) for i in top_indices]

    def predict(self, user_id, movie_id):
        """
        Predict rating for a user-movie pair using weighted average
        of similar items the user has rated.
        """
        if movie_id not in self.movie_index:
            return self.global_mean

        if user_id not in self.user_means.index:
            return self.global_mean

        similar_items = self.get_similar_items(movie_id, n=self.n_similar)
        user_row = self.raw_ratings.loc[user_id] if user_id in self.raw_ratings.index.get_level_values(0) else None

        numerator = 0
        denominator = 0
        for similar_movie, similarity in similar_items:
            try:
                rating = self.raw_ratings.loc[(user_id, similar_movie)]
                numerator += similarity * rating
                denominator += abs(similarity)
            except KeyError:
                continue

        if denominator == 0:
            return self.global_mean

        return numerator / denominator

    def recommend(self, user_id, seen_movies: set, n=10):
        """
        Recommend top-N unseen movies for a user.
        """
        if user_id not in self.user_means.index:
            return []

        candidate_movies = [
            m for m in self.eligible_movies
            if m not in seen_movies
        ]

        predictions = [
            (movie_id, self.predict(user_id, movie_id))
            for movie_id in candidate_movies
        ]
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [movie_id for movie_id, _ in predictions[:n]]

    def recommend_all(self, test_ratings, train_ratings, n=10, sample_users=None):
        """
        Generate recommendations for all users using matrix multiplication.
        Fully vectorized top-N selection using numpy.
    
        Args:
            sample_users: if provided, randomly sample this many users for evaluation
        """
        # Sample users if requested
        all_test_users = test_ratings['userId'].unique()
        if sample_users and sample_users < len(all_test_users):
            np.random.seed(42)  # reproducibility
            sampled = np.random.choice(all_test_users, size=sample_users, replace=False)
            test_ratings = test_ratings[test_ratings['userId'].isin(sampled)]
            print(f"Sampled {sample_users} users from {len(all_test_users)} total")
            print("Building user-item matrix for scoring...")

        filtered = train_ratings[
            train_ratings['movieId'].isin(self.eligible_movies)
        ].copy()

        test_users = test_ratings['userId'].unique()
        test_user_index = {uid: idx for idx, uid in enumerate(test_users)}

        user_means = filtered.groupby('userId')['rating'].mean()
        filtered['rating_centered'] = (
            filtered['rating'] - filtered['userId'].map(user_means)
        )

        filtered_test = filtered[filtered['userId'].isin(test_users)].copy()

        rows = filtered_test['userId'].map(test_user_index).values
        cols = filtered_test['movieId'].map(self.movie_index).values
        data = filtered_test['rating_centered'].values

        valid = ~pd.isna(cols)
        rows = rows[valid]
        cols = cols[valid].astype(int)
        data = data[valid]

        user_item_sparse = csr_matrix(
            (data, (rows, cols)),
            shape=(len(test_users), len(self.movie_index))
        )

        print("Computing scores via matrix multiplication...")
        # Convert to dense immediately — indexing dense arrays is much faster
        scores_dense = (user_item_sparse @ self.item_similarity)
        if hasattr(scores_dense, 'toarray'):
            scores_dense = scores_dense.toarray()

        print("Building seen movies mask...")
        user_train_movies = (
            train_ratings[train_ratings['movieId'].isin(self.eligible_movies)]
            .groupby('userId')['movieId']
            .apply(set)
            .to_dict()
        )

        all_movie_ids = np.array([self.index_to_movie[i] for i in range(len(self.movie_index))])

        print("Generating top-N recommendations per user...")
        recommendations = {}
        for i, user_id in enumerate(test_users):
            if i % 10000 == 0:
                print(f"  {i}/{len(test_users)} users processed...")

            user_scores = scores_dense[test_user_index[user_id]].copy()

            # Mask seen movies with -inf — no filtering loop needed
            seen = user_train_movies.get(user_id, set())
            seen_indices = [
                self.movie_index[m] for m in seen
                if m in self.movie_index
            ]
            user_scores[seen_indices] = -np.inf

            # argpartition finds top-N without full sort — O(n) not O(n log n)
            top_indices = np.argpartition(user_scores, -n)[-n:]
            top_indices = top_indices[np.argsort(user_scores[top_indices])[::-1]]
            recommendations[user_id] = all_movie_ids[top_indices].tolist()

        return recommendations