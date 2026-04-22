import sys
from dotenv import load_dotenv
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from neo4j import GraphDatabase

load_dotenv()

# --- Config ---
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

TRAIN_PATH = "data/splits/train_ratings.csv"
MOVIES_PATH = "data/raw/movie.csv"
GENOME_SCORES_PATH = "data/raw/genome_scores.csv"
GENOME_TAGS_PATH = "data/raw/genome_tags.csv"

TOP_USERS = 10000
TOP_MOVIES = 5000
RELEVANCE_THRESHOLD = 0.6


def get_subset(train):
    """Select top users and movies by activity."""
    print("Selecting top users and movies...")
    
    top_users = (
        train.groupby('userId').size()
        .sort_values(ascending=False)
        .head(TOP_USERS)
        .index
    )
    
    top_movies = (
        train.groupby('movieId').size()
        .sort_values(ascending=False)
        .head(TOP_MOVIES)
        .index
    )
    
    subset = train[
        train['userId'].isin(top_users) &
        train['movieId'].isin(top_movies)
    ]
    
    print(f"Subset: {len(subset)} ratings")
    print(f"Users: {subset['userId'].nunique()}")
    print(f"Movies: {subset['movieId'].nunique()}")
    
    return subset, top_users, top_movies


def clear_database(session):
    print("Clearing existing data...")
    session.run("MATCH (n) DETACH DELETE n")


def create_constraints(session):
    print("Creating constraints...")
    session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.userId IS UNIQUE")
    session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (m:Movie) REQUIRE m.movieId IS UNIQUE")
    session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (g:Genre) REQUIRE g.name IS UNIQUE")
    session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Tag) REQUIRE t.tagId IS UNIQUE")


def load_users(session, user_ids):
    print(f"Loading {len(user_ids)} users...")
    batch_size = 1000
    user_list = [{'userId': int(uid)} for uid in user_ids]
    
    for i in range(0, len(user_list), batch_size):
        batch = user_list[i:i+batch_size]
        session.run("""
            UNWIND $users AS user
            MERGE (u:User {userId: user.userId})
        """, users=batch)
    print("Users loaded.")


def load_movies(session, movie_ids, movies_df):
    print(f"Loading {len(movie_ids)} movies...")
    movies_subset = movies_df[movies_df['movieId'].isin(movie_ids)]
    batch_size = 1000
    
    movie_list = [
        {'movieId': int(row['movieId']), 'title': row['title']}
        for _, row in movies_subset.iterrows()
    ]
    
    for i in range(0, len(movie_list), batch_size):
        batch = movie_list[i:i+batch_size]
        session.run("""
            UNWIND $movies AS movie
            MERGE (m:Movie {movieId: movie.movieId})
            SET m.title = movie.title
        """, movies=batch)
    print("Movies loaded.")


def load_genres(session, movie_ids, movies_df):
    print("Loading genres...")
    movies_subset = movies_df[movies_df['movieId'].isin(movie_ids)].copy()
    
    # Explode genres
    movies_subset['genres'] = movies_subset['genres'].str.split('|')
    movies_subset = movies_subset.explode('genres')
    movies_subset = movies_subset[movies_subset['genres'] != '(no genres listed)']
    
    genre_data = [
        {'movieId': int(row['movieId']), 'genre': row['genres']}
        for _, row in movies_subset.iterrows()
    ]
    
    batch_size = 1000
    for i in range(0, len(genre_data), batch_size):
        batch = genre_data[i:i+batch_size]
        session.run("""
            UNWIND $data AS d
            MATCH (m:Movie {movieId: d.movieId})
            MERGE (g:Genre {name: d.genre})
            MERGE (m)-[:HAS_GENRE]->(g)
        """, data=batch)
    print("Genres loaded.")


def load_tags(session, movie_ids, genome_scores, genome_tags):
    print("Loading genome tags...")
    
    # Filter to eligible movies and threshold
    filtered = genome_scores[
        (genome_scores['movieId'].isin(movie_ids)) &
        (genome_scores['relevance'] >= RELEVANCE_THRESHOLD)
    ]
    
    # Merge with tag names
    filtered = filtered.merge(genome_tags, on='tagId')
    
    tag_data = [
        {
            'movieId': int(row['movieId']),
            'tagId': int(row['tagId']),
            'tag': row['tag'],
            'relevance': float(row['relevance'])
        }
        for _, row in filtered.iterrows()
    ]
    
    batch_size = 1000
    total = len(tag_data)
    for i in range(0, total, batch_size):
        if i % 50000 == 0:
            print(f"  Tags: {i}/{total}...")
        batch = tag_data[i:i+batch_size]
        session.run("""
            UNWIND $data AS d
            MATCH (m:Movie {movieId: d.movieId})
            MERGE (t:Tag {tagId: d.tagId})
            SET t.name = d.tag
            MERGE (m)-[:HAS_TAG {relevance: d.relevance}]->(t)
        """, data=batch)
    print("Tags loaded.")


def load_ratings(session, subset):
    print(f"Loading {len(subset)} ratings...")
    batch_size = 1000
    
    rating_data = [
        {
            'userId': int(row['userId']),
            'movieId': int(row['movieId']),
            'rating': float(row['rating']),
            'timestamp': int(pd.Timestamp(row['timestamp']).timestamp())
        }
        for _, row in subset.iterrows()
    ]
    
    total = len(rating_data)
    for i in range(0, total, batch_size):
        if i % 100000 == 0:
            print(f"  Ratings: {i}/{total}...")
        batch = rating_data[i:i+batch_size]
        session.run("""
            UNWIND $data AS d
            MATCH (u:User {userId: d.userId})
            MATCH (m:Movie {movieId: d.movieId})
            MERGE (u)-[:RATED {rating: d.rating, timestamp: d.timestamp}]->(m)
        """, data=batch)
    print("Ratings loaded.")


def main():
    print("Loading data...")
    train = pd.read_csv(TRAIN_PATH)
    movies = pd.read_csv(MOVIES_PATH)
    genome_scores = pd.read_csv(GENOME_SCORES_PATH)
    genome_tags = pd.read_csv(GENOME_TAGS_PATH)

    subset, top_users, top_movies = get_subset(train)

    print("\nConnecting to Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        clear_database(session)
        create_constraints(session)
        load_users(session, top_users)
        load_movies(session, top_movies, movies)
        load_genres(session, top_movies, movies)
        load_tags(session, top_movies, genome_scores, genome_tags)
        load_ratings(session, subset)

    driver.close()
    print("\nGraph loaded successfully.")


if __name__ == "__main__":
    main()