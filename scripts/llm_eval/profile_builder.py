import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np


# Load data once at module level
_train = None
_movies = None
_genome_scores = None
_genome_tags = None


def _load_data():
    global _train, _movies, _genome_scores, _genome_tags
    if _train is None:
        _train = pd.read_csv('data/splits/train_ratings.csv')
        _movies = pd.read_csv('data/raw/movie.csv')
        _genome_scores = pd.read_csv('data/raw/genome_scores.csv')
        _genome_tags = pd.read_csv('data/raw/genome_tags.csv')


def build_taste_profile(user_id: int) -> dict:
    """
    Build a compact taste profile for a user from their rating history.
    Designed to fit in LLM context window — ~100 tokens max.
    
    Returns dict with:
        - top_genres: list of top 3 genres
        - top_tags: list of top 5 genome tags
        - favourite_movies: list of top 5 movie titles
        - avg_rating: float
        - total_ratings: int
        - rating_style: str (generous/balanced/critical)
    """
    _load_data()

    user_ratings = _train[_train['userId'] == user_id]
    if user_ratings.empty:
        return None

    # Basic stats
    avg_rating = user_ratings['rating'].mean()
    total_ratings = len(user_ratings)

    # Rating style
    if avg_rating >= 4.0:
        rating_style = "generous rater"
    elif avg_rating >= 3.2:
        rating_style = "balanced rater"
    else:
        rating_style = "critical rater"

    # Top genres from highly rated movies
    liked = user_ratings[user_ratings['rating'] >= 4.0]
    if liked.empty:
        liked = user_ratings.nlargest(20, 'rating')

    liked_movies = liked.merge(_movies, on='movieId')
    genres = liked_movies['genres'].str.split('|').explode()
    genres = genres[genres != '(no genres listed)']
    top_genres = genres.value_counts().head(3).index.tolist()

    # Top genome tags from liked movies
    liked_movie_ids = liked['movieId'].tolist()
    tag_scores = _genome_scores[
        (_genome_scores['movieId'].isin(liked_movie_ids)) &
        (_genome_scores['relevance'] >= 0.5)
    ]
    if not tag_scores.empty:
        avg_tag_scores = tag_scores.groupby('tagId')['relevance'].mean()
        top_tag_ids = avg_tag_scores.nlargest(5).index.tolist()
        top_tags = _genome_tags[
            _genome_tags['tagId'].isin(top_tag_ids)
        ]['tag'].tolist()
    else:
        top_tags = []

    # Favourite movies
    top_movies = (
        user_ratings.nlargest(5, 'rating')
        .merge(_movies, on='movieId')['title']
        .tolist()
    )
    # Clean titles
    top_movies = [_clean_title(t) for t in top_movies]

    return {
        'user_id': user_id,
        'total_ratings': total_ratings,
        'avg_rating': round(avg_rating, 2),
        'rating_style': rating_style,
        'top_genres': top_genres,
        'top_tags': top_tags,
        'favourite_movies': top_movies
    }


def profile_to_text(profile: dict) -> str:
    """
    Convert taste profile dict to compact text for LLM prompt.
    """
    return f"""User taste profile:
- Favourite genres: {', '.join(profile['top_genres']) if profile['top_genres'] else 'varied'}
- Key themes/tags: {', '.join(profile['top_tags']) if profile['top_tags'] else 'varied'}
- Favourite movies: {', '.join(profile['favourite_movies'])}
- Rating style: {profile['rating_style']} (avg {profile['avg_rating']}/5.0, {profile['total_ratings']} total ratings)"""


def _clean_title(title: str) -> str:
    """Move trailing articles to front."""
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


def get_eligible_users(n: int = 50, seed: int = 42) -> list:
    """
    Get users that are eligible for evaluation:
    - Present in train set
    - In Neo4j graph (top 10K most active users)
    - Have at least 50 ratings
    
    Returns list of n user_ids.
    """
    _load_data()

    # Get top 10K most active users (graph subset)
    user_counts = _train.groupby('userId').size()
    top_10k = user_counts.nlargest(10000).index

    # Filter to users with at least 50 ratings
    eligible = user_counts[
        (user_counts >= 50) &
        (user_counts.index.isin(top_10k))
    ].index.tolist()

    # Sample n users
    np.random.seed(seed)
    sampled = np.random.choice(eligible, size=min(n, len(eligible)), replace=False)
    return [int(u) for u in sampled.tolist()]