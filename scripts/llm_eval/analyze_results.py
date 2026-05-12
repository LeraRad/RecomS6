import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import json
import pandas as pd
import numpy as np


RESULTS_FILE = "results/llm_eval_results.json"
MODELS_ORDER = ["SVD", "Item-CF", "ALS", "LightFM", "Graph", "Popularity"]


def load_results() -> pd.DataFrame:
    with open(RESULTS_FILE, 'r') as f:
        results = json.load(f)
    return pd.DataFrame(results)


def analyze():
    if not os.path.exists(RESULTS_FILE):
        print(f"No results file found at {RESULTS_FILE}")
        print("Run run_evaluation.py first.")
        return

    df = load_results()
    df = df[df['score'].notna()]

    print(f"Total evaluations: {len(df)}")
    print(f"Users evaluated: {df['user_id'].nunique()}")
    print(f"Models evaluated: {df['model'].nunique()}")
    print()

    # --- Per-model summary ---
    print("=" * 60)
    print("LLM EVALUATION RESULTS — Average Score per Model (1-10)")
    print("=" * 60)

    summary = df.groupby('model')['score'].agg(
        avg_score='mean',
        median_score='median',
        std_score='std',
        n_evaluations='count'
    ).round(3)

    # Sort by average score
    summary = summary.sort_values('avg_score', ascending=False)

    # Print table
    print(f"\n{'Model':<15} {'Avg Score':>10} {'Median':>10} {'Std':>8} {'N':>6}")
    print("-" * 55)
    for model, row in summary.iterrows():
        print(f"{model:<15} {row['avg_score']:>10.2f} {row['median_score']:>10.2f} "
              f"{row['std_score']:>8.2f} {int(row['n_evaluations']):>6}")

    # --- Score distribution ---
    print("\n--- Score Distribution ---")
    score_dist = df.groupby('model')['score'].value_counts().unstack(fill_value=0)
    score_dist = score_dist.reindex(
        columns=range(1, 11), fill_value=0
    )
    print(score_dist)

    # --- Sample reasoning per model ---
    print("\n--- Sample Evaluations (highest scoring per model) ---")
    for model in MODELS_ORDER:
        model_df = df[df['model'] == model].sort_values('score', ascending=False)
        if model_df.empty:
            continue
        best = model_df.iloc[0]
        print(f"\n{model} — Best score: {best['score']}/10")
        print(f"  {best['reasoning']}")

    # --- Common strengths and weaknesses ---
    print("\n--- Most Common Strengths ---")
    all_strengths = []
    for strengths in df['strengths'].dropna():
        if isinstance(strengths, list):
            all_strengths.extend(strengths)
    from collections import Counter
    strength_counts = Counter(all_strengths).most_common(10)
    for strength, count in strength_counts:
        print(f"  {count:>3}x — {strength}")

    print("\n--- Most Common Weaknesses ---")
    all_weaknesses = []
    for weaknesses in df['weaknesses'].dropna():
        if isinstance(weaknesses, list):
            all_weaknesses.extend(weaknesses)
    weakness_counts = Counter(all_weaknesses).most_common(10)
    for weakness, count in weakness_counts:
        print(f"  {count:>3}x — {weakness}")

    # --- Save summary to JSON ---
    summary_output = {
        'total_evaluations': len(df),
        'users_evaluated': int(df['user_id'].nunique()),
        'model_scores': summary['avg_score'].to_dict(),
        'evaluation_model': 'llama3.1:8b'
    }

    summary_path = "results/llm_eval_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary_output, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    analyze()