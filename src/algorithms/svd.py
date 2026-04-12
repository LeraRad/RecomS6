import pickle
import numpy as np
import pandas as pd
from surprise import SVD, Dataset, Reader

class SVDRecommender:
    """
    SVD-based collaborative filtering recommender.
    Uses matrix factorization to discover hidden user and movie features.
    Output: ranked list of (movie_id, score) tuples — unified interface.
    """
    
    def __init__(self, n_factors=50, n_epochs=20, biased=True):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.biased = biased
        self.model = None
        self.trainset = None

    def train(self, train_ratings: pd.DataFrame):
        """
        Train SVD model on ratings dataframe.
        
        Args:
            train_ratings: DataFrame with columns [userId, movieId, rating]
        """
        reader = Reader(rating_scale=(
            train_ratings['rating'].min(), 
            train_ratings['rating'].max()
        ))
        
        data = Dataset.load_from_df(
            train_ratings[['userId', 'movieId', 'rating']], 
            reader
        )
        
        self.trainset = data.build_full_trainset()
        
        self.model = SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            biased=self.biased,
            verbose=True
        )
        
        self.model.fit(self.trainset)
        print("SVD training complete!")

    def recommend(self, user_id: int, n: int = 10) -> list:
        """
        Generate top-N recommendations for a user.
        
        Args:
            user_id: ID of the user
            n: number of recommendations to return
            
        Returns:
            List of (movie_id, score) tuples sorted by score descending
        """
        # Get all movies user hasn't rated yet
        all_movie_ids = self.trainset.all_items()
        rated_movies = set(
            self.trainset.ur[self.trainset.to_inner_uid(user_id)]
            .keys()
        )
        
        # Predict scores for unseen movies
        predictions = []
        for movie_inner_id in all_movie_ids:
            if movie_inner_id not in rated_movies:
                movie_id = self.trainset.to_raw_iid(movie_inner_id)
                score = self.model.predict(user_id, movie_id).est
                predictions.append((movie_id, score))
        
        # Sort and return top-N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n]
    
    def save(self, path: str):
        """Save trained model to disk."""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load trained model from disk."""
        with open(path, 'rb') as f:
            loaded = pickle.load(f)
        self.model = loaded.model
        self.trainset = loaded.trainset
        print(f"Model loaded from {path}")