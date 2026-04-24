import requests
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("TMDB_API_KEY")

# Test fetch — The Matrix (tmdbId: 603)
response = requests.get(
    f"https://api.themoviedb.org/3/movie/603",
    params={"api_key": API_KEY}
)
print(response.json().get("title"))
print(response.json().get("poster_path"))

links = pd.read_csv('data/raw/link.csv')
print(links.head())
print(links.dtypes)