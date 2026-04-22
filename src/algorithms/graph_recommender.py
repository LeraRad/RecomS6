import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os
from neo4j import GraphDatabase


NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


class GraphRecommender:
    """
    Graph-based recommender using Neo4j.
    Combines collaborative filtering with content-based signals
    via graph traversal through user ratings, movie genres and genome tags.
    """

    def __init__(self, uri=None, user=None, password=None):
        load_dotenv()
        uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = user or os.getenv("NEO4J_USER", "neo4j")
        password = password or os.getenv("NEO4J_PASSWORD")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.user_ids_in_graph = None

    def train(self, train_ratings: pd.DataFrame):
        """
        No training needed — graph is pre-built.
        Just verify connection and store which users are in graph.
        """
        print("Connecting to Neo4j...")
        with self.driver.session() as session:
            result = session.run("MATCH (u:User) RETURN count(u) AS count")
            count = result.single()['count']
            print(f"Connected. Users in graph: {count}")

            # Store which userIds exist in graph
            result = session.run("MATCH (u:User) RETURN u.userId AS userId")
            self.user_ids_in_graph = set(record['userId'] for record in result)

        print("Graph recommender ready.")

    def _recommend_for_user(self, session, user_id, n=10):
        """
        Generate recommendations for a single user via graph traversal.
        
        Strategy:
        1. Find similar users via shared highly-rated movies
        2. Find candidate movies from similar users
        3. Boost candidates that share high-relevance tags with user's liked movies
        4. Return top-N scored candidates
        """
        result = session.run("""
            // Step 1: Find movies target user liked (limit to most recent 50)
            MATCH (target:User {userId: $user_id})-[r:RATED]->(liked:Movie)
            WHERE r.rating >= 3.5
            WITH target, liked, r
            ORDER BY r.timestamp DESC
            LIMIT 50
            
            // Step 2: Find similar users who also liked those movies
            WITH target, liked
            MATCH (liked)<-[r2:RATED]-(similar:User)
            WHERE r2.rating >= 3.5 AND similar.userId <> $user_id
            
            WITH target, similar, count(DISTINCT liked) AS shared_movies
            ORDER BY shared_movies DESC
            LIMIT 10
            
            // Step 3: Find candidate movies from similar users
            MATCH (similar)-[r3:RATED]->(candidate:Movie)
            WHERE r3.rating >= 3.5
            AND NOT (target)-[:RATED]->(candidate)
            
            // Step 4: Boost by tag overlap with user's liked movies
            WITH target, candidate, count(similar) AS cf_score
            OPTIONAL MATCH (target)-[r4:RATED]->(liked2:Movie)-[:HAS_TAG]->(t:Tag)<-[:HAS_TAG]-(candidate)
            WHERE r4.rating >= 3.5
            
            WITH candidate,
                cf_score,
                count(DISTINCT t) AS tag_overlap
                 
            // Combined score: collaborative signal + content signal
            WITH candidate,
                 cf_score + (tag_overlap * 0.5) AS final_score
                 
            RETURN candidate.movieId AS movieId, final_score
            ORDER BY final_score DESC
            LIMIT $n
        """, user_id=user_id, n=n)

        return [record['movieId'] for record in result]

    def recommend_all(self, test_ratings, train_ratings, n=10, sample_users=None):
        """
        Generate recommendations for all users in test set.
        """
        test_users = test_ratings['userId'].unique()

        if sample_users and sample_users < len(test_users):
            np.random.seed(42)
            test_users = np.random.choice(test_users, size=sample_users, replace=False)
            print(f"Sampled {sample_users} users from {len(test_users)} total")

        # Only recommend for users that exist in graph
        test_users = [uid for uid in test_users if uid in self.user_ids_in_graph]
        print(f"Users found in graph: {len(test_users)}")

        recommendations = {}
        total = len(test_users)

        with self.driver.session() as session:
            for i, user_id in enumerate(test_users):
                if i % 500 == 0:
                    print(f"  {i}/{total} users processed...")
                recommendations[user_id] = self._recommend_for_user(
                    session, int(user_id), n
                )

        return recommendations

    def close(self):
        self.driver.close()