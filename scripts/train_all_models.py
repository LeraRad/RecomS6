import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

print("Loading data...")
train = pd.read_csv('data/splits/train_ratings.csv')
print(f"Train: {len(train)} ratings")

# Create models directory
os.makedirs('models', exist_ok=True)

# --- SVD ---
print("\nTraining SVD...")
from src.algorithms.svd import SVDRecommender
svd = SVDRecommender(n_factors=50, n_epochs=20, biased=True)
svd.train(train)
svd.save('models/svd.pkl')

# --- Item-CF ---
print("\nTraining Item-CF...")
from src.algorithms.item_cf import ItemCFRecommender
item_cf = ItemCFRecommender(min_ratings=50, n_similar=20)
item_cf.train(train)
item_cf.save('models/item_cf.pkl')

# --- ALS ---
print("\nTraining ALS...")
from src.algorithms.als import ALSRecommender
als = ALSRecommender(factors=100, iterations=30, alpha=20)
als.train(train)
als.save('models/als.pkl')

# --- LightFM ---
print("\nTraining LightFM...")
from src.algorithms.lightfm_recommender import LightFMRecommender
from src.data.feature_engineering import build_movie_features

eligible_movies = set(train['movieId'].unique())
item_features, movie_index, tag_index = build_movie_features(
    'data/raw/genome_scores.csv',
    'data/raw/genome_tags.csv',
    eligible_movies,
    relevance_threshold=0.6
)
lightfm = LightFMRecommender(no_components=50, loss='warp', epochs=30)
lightfm.train(train, item_features_matrix=item_features, movie_index=movie_index)
lightfm.save('models/lightfm.pkl')

# --- Popularity ---
print("\nTraining Popularity...")
from src.algorithms.popularity import PopularityRecommender
popularity = PopularityRecommender()
popularity.train(train)
popularity.save('models/popularity.pkl')

print("\nAll models trained and saved to models/")
print("Files:")
for f in os.listdir('models'):
    size = os.path.getsize(f'models/{f}') / 1024 / 1024
    print(f"  {f}: {size:.1f} MB")