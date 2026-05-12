import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from scripts.llm_eval.profile_builder import build_taste_profile, profile_to_text, get_eligible_users
from scripts.llm_eval.evaluator import evaluate_recommendations
from app.recommender import get_recommendations

# Get one eligible user
print("Getting eligible users...")
users = get_eligible_users(n=5, seed=42)
user_id = users[0]
print(f"Testing with user {user_id}")

# Build profile
print("\nBuilding taste profile...")
profile = build_taste_profile(user_id)
profile_text = profile_to_text(profile)
print(profile_text)

# Get recommendations from one model
print("\nGetting LightFM recommendations...")
movie_ids = get_recommendations(user_id, 'LightFM', n=10)
print(f"Got {len(movie_ids)} recommendations")

# Evaluate
print("\nEvaluating with Ollama...")
result = evaluate_recommendations(profile_text, movie_ids)
print(f"\nScore: {result['score']}/10")
print(f"Reasoning: {result['reasoning']}")
print(f"Strengths: {result['strengths']}")
print(f"Weaknesses: {result['weaknesses']}")