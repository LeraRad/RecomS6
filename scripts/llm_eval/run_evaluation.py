import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import json
import time
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from scripts.llm_eval.profile_builder import build_taste_profile, profile_to_text, get_eligible_users
from scripts.llm_eval.evaluator import evaluate_recommendations

# --- Config ---
N_USERS = 50
SEED = 42
OLLAMA_MODEL = "llama3.1:8b"
RESULTS_DIR = "results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "llm_eval_results.json")

MODELS_TO_EVALUATE = ["SVD", "Item-CF", "ALS", "LightFM", "Popularity", "Graph"]


def get_recommendations_for_user(user_id: int, model_name: str, train: pd.DataFrame) -> list:
    """Load model and generate recommendations for a single user."""
    from app.recommender import get_recommendations
    return get_recommendations(user_id, model_name, n=10)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load existing results if any — allows resuming interrupted runs
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing results")
    else:
        results = []

    # Get eligible users
    print(f"Selecting {N_USERS} evaluation users...")
    user_ids = get_eligible_users(n=N_USERS, seed=SEED)
    print(f"Selected {len(user_ids)} users")

    # Track completed user-model pairs
    completed = set(
        (r['user_id'], r['model']) for r in results
    )

    total = len(user_ids) * len(MODELS_TO_EVALUATE)
    done = len(completed)
    print(f"Progress: {done}/{total} evaluations completed\n")

    for user_id in user_ids:
        # Build taste profile once per user
        profile = build_taste_profile(user_id)
        if profile is None:
            print(f"Skipping user {user_id} — no profile data")
            continue

        profile_text = profile_to_text(profile)

        for model_name in MODELS_TO_EVALUATE:
            if (user_id, model_name) in completed:
                print(f"Skipping user {user_id} / {model_name} — already evaluated")
                continue

            print(f"Evaluating user {user_id} / {model_name}...")

            try:
                # Get recommendations
                movie_ids = get_recommendations_for_user(user_id, model_name, None)

                if not movie_ids:
                    print(f"  No recommendations generated — skipping")
                    continue

                # Evaluate with Ollama
                eval_result = evaluate_recommendations(
                    profile_text=profile_text,
                    movie_ids=movie_ids,
                    model_name=OLLAMA_MODEL
                )

                # Store result
                result_entry = {
                    'user_id': int(user_id),  # convert from numpy int64
                    'model': model_name,
                    'score': eval_result['score'],
                    'reasoning': eval_result['reasoning'],
                    'strengths': eval_result['strengths'],
                    'weaknesses': eval_result['weaknesses'],
                    'profile_summary': profile_text,
                    'recommended_movies': [int(m) for m in movie_ids]  # convert list too
                }
                results.append(result_entry)
                completed.add((user_id, model_name))

                print(f"  Score: {eval_result['score']}/10")
                print(f"  {eval_result['reasoning'][:100]}...")

                # Save after every evaluation — safe against interruptions
                with open(RESULTS_FILE, 'w') as f:
                    json.dump(results, f, indent=2)

                # Small delay to not overwhelm Ollama
                time.sleep(0.5)

            except Exception as e:
                print(f"  Error: {e}")
                continue

    print(f"\nEvaluation complete. Results saved to {RESULTS_FILE}")
    print(f"Total evaluations: {len(results)}")


if __name__ == "__main__":
    main()