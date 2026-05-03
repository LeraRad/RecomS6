import requests
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

# Load links once at module level
_links = None

def get_links():
    global _links
    if _links is None:
        _links = pd.read_csv('data/raw/link.csv')
    return _links


def get_poster_url(movie_id: int) -> str:
    """
    Get TMDB poster URL for a MovieLens movieId.
    Returns None if poster not found.
    """
    links = get_links()
    match = links[links['movieId'] == movie_id]
    
    if match.empty or pd.isna(match.iloc[0]['tmdbId']):
        return None
    
    tmdb_id = int(match.iloc[0]['tmdbId'])
    
    try:
        response = requests.get(
            f"{TMDB_BASE_URL}/movie/{tmdb_id}",
            params={"api_key": TMDB_API_KEY},
            timeout=5
        )
        data = response.json()
        poster_path = data.get("poster_path")
        if poster_path:
            return f"{TMDB_IMAGE_BASE}{poster_path}"
    except Exception:
        return None
    
    return None


def get_movie_details(movie_id: int) -> dict:
    """
    Get movie title and poster URL for a MovieLens movieId.
    """
    links = get_links()
    match = links[links['movieId'] == movie_id]
    
    if match.empty or pd.isna(match.iloc[0]['tmdbId']):
        return {'poster_url': None, 'overview': ''}
    
    tmdb_id = int(match.iloc[0]['tmdbId'])
    
    try:
        response = requests.get(
            f"{TMDB_BASE_URL}/movie/{tmdb_id}",
            params={"api_key": TMDB_API_KEY},
            timeout=5
        )
        data = response.json()
        poster_path = data.get("poster_path")
        return {
            'poster_url': f"{TMDB_IMAGE_BASE}{poster_path}" if poster_path else None,
            'overview': data.get("overview", "")
        }
    except Exception:
        return {'poster_url': None, 'overview': ''}