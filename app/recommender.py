import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import pickle
from dotenv import load_dotenv

load_dotenv()

# --- Load data once ---
print("Loading data...")
train = pd.read_csv('data/splits/train_ratings.csv')
test = pd.read_csv('data/splits/test_ratings.csv')
movies = pd.read_csv('data/raw/movie.csv')

# Movie title lookup
movie_titles = movies.set_index('movieId')['title'].to_dict()

def get_movie_title(movie_id: int) -> str:
    title = movie_titles.get(movie_id, f"Movie {movie_id}")
    return clean_title(title)

def clean_title(title: str) -> str:
    """Move trailing articles back to front. 'Matrix, The (1999)' → 'The Matrix (1999)'"""
    for article in [', The', ', A', ', An']:
        if article in title:
            year = ''
            if title.endswith(')'):
                year_start = title.rfind('(')
                year = ' ' + title[year_start:]
                title = title[:year_start].strip()
            title = title.replace(article, '')
            return article.strip(', ') + ' ' + title.strip() + year
    return title

def get_user_profile(user_id: int) -> dict:
    """
    Build a summary of user's taste from their rating history.
    """
    user_ratings = train[train['userId'] == user_id]
    
    if user_ratings.empty:
        return None
    
    # Basic stats
    total_ratings = len(user_ratings)
    avg_rating = user_ratings['rating'].mean()
    
    # Top genres
    user_movies = user_ratings.merge(movies, on='movieId')
    genres = user_movies['genres'].str.split('|').explode()
    top_genres = genres.value_counts().head(3).index.tolist()
    
    # Favorite movies (highest rated)
    top_movies = (
        user_ratings.nlargest(3, 'rating')
        [['movieId', 'rating']]
        .merge(movies, on='movieId')
        ['title']
        .apply(clean_title)
        .tolist()
    )
    
    # Rating era
    user_ratings = user_ratings.copy()
    user_ratings['year'] = pd.to_datetime(
        user_ratings['timestamp'], unit='s', errors='coerce'
    ).dt.year
    most_active_year = user_ratings['year'].mode()
    most_active_year = int(most_active_year.iloc[0]) if not most_active_year.empty else None

    return {
        'total_ratings': total_ratings,
        'avg_rating': round(avg_rating, 2),
        'top_genres': top_genres,
        'top_movies': top_movies,
        'most_active_year': most_active_year
    }

def get_closest_graph_user(user_id: int, graph_user_ids: set) -> int:
    graph_ids = sorted(graph_user_ids)
    # Find closest ID numerically
    closest = min(graph_ids, key=lambda x: abs(x - user_id))
    return int(closest)


def get_graph_user_ids() -> set:
    return get_graph().user_ids_in_graph

# --- Model cache ---
_models = {}
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

def get_svd():
    if 'svd' not in _models:
        from src.algorithms.svd import SVDRecommender
        path = os.path.join(MODELS_DIR, 'svd.pkl')
        if os.path.exists(path):
            print("Loading SVD from disk...")
            model = SVDRecommender()
            model.load(path)
        else:
            print("Training SVD...")
            model = SVDRecommender(n_factors=50, n_epochs=20, biased=True)
            model.train(train)
        _models['svd'] = model
    return _models['svd']

def get_item_cf():
    if 'item_cf' not in _models:
        from src.algorithms.item_cf import ItemCFRecommender
        path = os.path.join(MODELS_DIR, 'item_cf.pkl')
        if os.path.exists(path):
            import time
            print("Loading Item-CF from disk...")
            start = time.time()
            _models['item_cf'] = ItemCFRecommender.load(path)
            print(f"Item-CF loaded in {time.time() - start:.1f}s")
        else:
            print("Training Item-CF...")
            model = ItemCFRecommender(min_ratings=50, n_similar=20)
            model.train(train)
            _models['item_cf'] = model
    return _models['item_cf']

def get_als():
    if 'als' not in _models:
        from src.algorithms.als import ALSRecommender
        path = os.path.join(MODELS_DIR, 'als.pkl')
        if os.path.exists(path):
            print("Loading ALS from disk...")
            _models['als'] = ALSRecommender.load(path)
        else:
            print("Training ALS...")
            model = ALSRecommender(factors=100, iterations=30, alpha=20)
            model.train(train)
            _models['als'] = model
    return _models['als']

def get_lightfm():
    if 'lightfm' not in _models:
        from src.algorithms.lightfm_recommender import LightFMRecommender
        path = os.path.join(MODELS_DIR, 'lightfm.pkl')
        if os.path.exists(path):
            print("Loading LightFM from disk...")
            _models['lightfm'] = LightFMRecommender.load(path)
        else:
            print("Training LightFM...")
            from src.data.feature_engineering import build_movie_features
            eligible_movies = set(train['movieId'].unique())
            item_features, movie_index, tag_index = build_movie_features(
                'data/raw/genome_scores.csv',
                'data/raw/genome_tags.csv',
                eligible_movies,
                relevance_threshold=0.6
            )
            model = LightFMRecommender(no_components=50, loss='warp', epochs=30)
            model.train(train, item_features_matrix=item_features, movie_index=movie_index)
            _models['lightfm'] = model
    return _models['lightfm']

def get_popularity():
    if 'popularity' not in _models:
        from src.algorithms.popularity import PopularityRecommender
        path = os.path.join(MODELS_DIR, 'popularity.pkl')
        if os.path.exists(path):
            print("Loading Popularity from disk...")
            _models['popularity'] = PopularityRecommender.load(path)
        else:
            print("Training Popularity...")
            model = PopularityRecommender()
            model.train(train)
            _models['popularity'] = model
    return _models['popularity']

def get_graph():
    if 'graph' not in _models:
        from src.algorithms.graph_recommender import GraphRecommender
        print("Connecting to Neo4j...")
        model = GraphRecommender()
        model.train(train)
        _models['graph'] = model
    return _models['graph']


def get_recommendations(user_id: int, model_name: str, n: int = 10) -> list:
    """
    Generate top-N recommendations for a user using specified model.
    Returns list of movie_ids.
    """
    seen = set(train[train['userId'] == user_id]['movieId'])

    if model_name == 'SVD':
        model = get_svd()
        recs = model.recommend_faiss(user_id, n=n)
        return [mid for mid, _ in recs]

    elif model_name == 'Item-CF':
        model = get_item_cf()
        return model.recommend(user_id, seen, n=n)

    elif model_name == 'ALS':
        model = get_als()
        user_df = pd.DataFrame({'userId': [user_id]})
        recs = model.recommend_all(user_df, train, n=n)
        return recs.get(user_id, [])

    elif model_name == 'LightFM':
        model = get_lightfm()
        user_df = pd.DataFrame({'userId': [user_id]})
        recs = model.recommend_all(user_df, train, n=n)
        return recs.get(user_id, [])

    elif model_name == 'Popularity':
        model = get_popularity()
        return model.recommend(user_id, seen, n=n)

    elif model_name == 'Graph':
        model = get_graph()
        user_df = pd.DataFrame({'userId': [user_id]})
        graph_users = set(model.user_ids_in_graph)
        if user_id not in graph_users:
            return None
        recs = model.recommend_all(user_df, train, n=n)
        return recs.get(user_id, [])

    return []

# --- Mode 2 helpers ---

GENRE_TAGS = {
    'Action': 19, 'Adventure': 29, 'Animation': 64, 'Biography': 134,
    'Comedy': 230, 'Coming of Age': 235, 'Crime': 268, 'Documentary': 315,
    'Family': 374, 'Fantasy': 377, 'Heist': 498, 'Horror': 522,
    'Martial Arts': 631, 'Musical': 686, 'Mystery': 689, 'Noir': 711,
    'Psychological': 823, 'Road Movie': 854, 'Romance': 863, 'Sci-Fi': 887,
    'Slasher': 927, 'Space': 942, 'Spy': 960, 'Superhero': 989,
    'Thriller': 1025, 'Time Travel': 1028, 'Vampire': 1067, 'War': 1096,
    'Western': 1107, 'Zombie': 1127
}

VIBE_TAGS = {
    'Atmospheric': 86, 'Based on a book': 107, 'Dark': 285, 'Dystopia': 336,
    'Emotional': 348, 'Feel-good': 388, 'Funny': 417, 'Historical': 508,
    'Plot twist': 789, 'Suspense': 999, 'Thought-provoking': 1024, 'True story': 1047
}

# Load genome scores once
_genome_scores = None

def get_genome_scores():
    global _genome_scores
    if _genome_scores is None:
        _genome_scores = pd.read_csv('data/raw/genome_scores.csv')
    return _genome_scores

def get_popular_movies_by_tags(selected_tag_ids: list, n: int = 15, offset: int = 0) -> pd.DataFrame:
    """
    Find most popular movies matching selected tags.
    offset allows pagination — skip already shown movies.
    """
    genome = get_genome_scores()
    movies_df = pd.read_csv('data/raw/movie.csv')

    filtered = genome[genome['tagId'].isin(selected_tag_ids)]
    movie_relevance = filtered.groupby('movieId')['relevance'].mean().reset_index()
    movie_relevance.columns = ['movieId', 'relevance_score']

    movie_popularity = train.groupby('movieId').agg(
        avg_rating=('rating', 'mean'),
        rating_count=('rating', 'count')
    ).reset_index()

    result = movie_relevance.merge(movie_popularity, on='movieId')
    result = result.merge(movies_df[['movieId', 'title']], on='movieId')
    result = result[result['rating_count'] >= 100]

    result['combined_score'] = (
        result['relevance_score'] * 0.6 +
        (result['avg_rating'] / 5.0) * 0.3 +
        (result['rating_count'] / result['rating_count'].max()) * 0.1
    )

    result = result.sort_values('combined_score', ascending=False)
    result = result.iloc[offset:offset + n]
    result['title'] = result['title'].apply(clean_title)
    return result


def get_content_based_recommendations(liked_movie_ids: list, n: int = 10) -> list:
    """
    Find movies similar to liked movies using genome tag similarity.
    Returns list of movieIds.
    """
    genome = get_genome_scores()

    # Get tag profiles for liked movies
    liked_tags = genome[
        (genome['movieId'].isin(liked_movie_ids)) &
        (genome['relevance'] >= 0.5)
    ].groupby('tagId')['relevance'].mean()

    if liked_tags.empty:
        return []

    # Score all other movies by tag overlap
    all_movies = genome[
        (~genome['movieId'].isin(liked_movie_ids)) &
        (genome['tagId'].isin(liked_tags.index))
    ].copy()

    all_movies['weighted_relevance'] = all_movies['tagId'].map(liked_tags) * all_movies['relevance']
    movie_scores = all_movies.groupby('movieId')['weighted_relevance'].sum()

    # Filter to movies with enough ratings
    popular = train.groupby('movieId').size()
    popular = popular[popular >= 50].index
    movie_scores = movie_scores[movie_scores.index.isin(popular)]

    top_movies = movie_scores.sort_values(ascending=False).head(n)
    return top_movies.index.tolist()