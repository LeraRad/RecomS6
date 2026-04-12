import numpy as np
import pandas as pd


class PopularityRecommender:
    """
    Popularity-based recommender.
    Recommends globally most rated movies to every user.
    No personalization — serves as baseline for comparison.
    """

    def __init__(self, n=10):
        self.n = n
        self.popular_movies = None

    def train(self, train_ratings: pd.DataFrame):
        """
        Compute globally most rated movies from training data.

        Args:
            train_ratings: DataFrame with columns [userId, movieId, rating]
        """
        self.popular_movies = (
            train_ratings.groupby('movieId')
            .size()
            .sort_values(ascending=False)
            .index.tolist()
        )

    def recommend(self, user_id, seen_movies: set, n=None):
        """
        Recommend top-N popular movies the user hasn't seen.

        Args:
            user_id: target user
            seen_movies: set of movieIds already seen by user in train
            n: number of recommendations (defaults to self.n)
        """
        if self.popular_movies is None:
            raise ValueError("Model not trained. Call train() first.")

        n = n or self.n
        recommendations = [
            movie for movie in self.popular_movies
            if movie not in seen_movies
        ]
        return recommendations[:n]

    def recommend_all(self, test_ratings, train_ratings, n=None):
        """
        Generate recommendations for all users in test set.
        Returns dict of {user_id: [movie_id, ...]}
        """
        n = n or self.n
        user_train_movies = (
            train_ratings.groupby('userId')['movieId']
            .apply(set)
            .to_dict()
        )

        recommendations = {}
        for user_id in test_ratings['userId'].unique():
            seen = user_train_movies.get(user_id, set())
            recommendations[user_id] = self.recommend(user_id, seen, n)

        return recommendations