import pickle
import numpy as np
import faiss
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
            iid for iid, _ in self.trainset.ur[self.trainset.to_inner_uid(user_id)]
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
    
    def recommend_faiss(self, user_id: int, n: int = 10) -> list:
        """
        Generate top-N recommendations using FAISS index for fast retrieval.
        Uses FAISS for candidate retrieval, then applies full SVD scoring.
        """
        import faiss
        import numpy as np

        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        trainset = self.model.trainset

        try:
            inner_uid = trainset.to_inner_uid(user_id)
        except ValueError:
            return []

        # Extract factor matrices
        pu = self.model.pu.astype(np.float32)
        qi = self.model.qi.astype(np.float32)
        bu = self.model.bu
        bi = self.model.bi
        global_mean = trainset.global_mean

        # Build FAISS index on item factors
        n_factors = qi.shape[1]
        index = faiss.IndexFlatIP(n_factors)
        index.add(qi)

        # Search top candidates using dot product
        user_vec = pu[inner_uid].reshape(1, -1)
        search_k = min(trainset.n_items, n * 20)
        _, candidate_indices = index.search(user_vec, search_k)

        # Get seen movies
        rated_inner_ids = set(
            iid for iid, _ in trainset.ur[inner_uid]
        )

        # Score unseen candidates with full SVD formula
        results = []
        for inner_iid in candidate_indices[0]:
            if inner_iid in rated_inner_ids:
                continue
            score = (
                global_mean
                + bu[inner_uid]
                + bi[inner_iid]
                + np.dot(pu[inner_uid], qi[inner_iid])
            )
            movie_id = trainset.to_raw_iid(int(inner_iid))
            results.append((movie_id, score))
            if len(results) == n:
                break

        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
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