# RecomS6 — Movie Recommendation System

A portfolio-grade recommendation system comparing 6 algorithms on the MovieLens 20M dataset. Built to demonstrate practical ML engineering skills across algorithm design, evaluation rigor, and production-oriented decisions.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

RecomS6 implements and evaluates six recommendation approaches — from a simple popularity baseline to a Neo4j graph recommender with genome tag traversal — using a unified evaluation pipeline and a Streamlit UI with TMDB movie posters.

The project is structured as a serious ML engineering exercise, not a tutorial follow-along. Every algorithm choice, evaluation decision, and engineering tradeoff is documented and defensible.

---

## Demo

**Mode 1 — Existing User**: Enter a MovieLens user ID, select a model, and get 10 personalized recommendations with posters and synopses. A user profile (top genres, favourite movies, rating history) is displayed alongside.

**Mode 2 — New User Cold Start**: No user ID required. Select genres and vibes, pick up to 3 movies you've seen, and get content-based recommendations generated from genome tag similarity.

---

## Algorithms

| Model | Type | Optimized For | Key Library |
|-------|------|--------------|-------------|
| Popularity Baseline | Non-personalized | N/A | pandas |
| SVD | Matrix Factorization | Rating Prediction | scikit-surprise |
| Item-based CF | Memory-based CF | Rating Prediction | numpy, scipy |
| ALS | Implicit Feedback MF | Top-N Ranking | implicit |
| LightFM | Hybrid (CF + Content) | Top-N Ranking | lightfm |
| Neo4j Graph | Graph Traversal | Top-N Ranking | Neo4j, Cypher |

### Key Finding — Objective Mismatch

SVD achieves excellent RMSE (0.80, near state-of-the-art for MovieLens 20M) but loses to the popularity baseline on Top-N ranking metrics. This demonstrates that **a model must be evaluated on the metric matching its training objective** — rating prediction accuracy does not imply good ranking performance.

---

## Results

Evaluation uses per-user temporal splits (last 20% of each user's interactions by count). Top-N metrics computed at K=10, relevance threshold 4.0.

| Metric | SVD | Item-CF | ALS | LightFM | Graph | Popularity |
|--------|-----|---------|-----|---------|-------|------------|
| Precision@10 | 0.0332 | 0.0803 | 0.0504 | 0.0896 | **0.1705** | 0.0575 |
| Recall@10 | 0.0290 | 0.0821 | 0.0689 | **0.0858** | 0.0300 | 0.0603 |
| NDCG@10 | 0.0420 | 0.1084 | 0.0666 | 0.1127 | **0.1829** | 0.0758 |
| RMSE | 0.7954 | N/A | N/A | N/A | N/A | N/A |

**Notes:**
- Item-CF, ALS, and LightFM evaluated on a random sample of 10,000 users (seed=42) for computational feasibility. Results are statistically representative.
- Graph recommender evaluated on 200 users from the top 10K most active users (the graph subset).
- Graph shows the highest Precision and NDCG but lowest Recall — a deliberate high-precision/low-recall tradeoff from staying close to the user's known taste neighborhood during traversal.
- LightFM benefits from both WARP ranking loss and movie genome tag features, making it the strongest generalizable model.

---

## Engineering Decisions

**Per-user temporal split** — last 20% of each user's interactions by count, sorted by timestamp. Prevents data leakage vs random split. Count-based rather than time-span-based to guarantee every user has a meaningful test set regardless of activity density.

**Sparse matrix for Item-CF** — dense pivot table would require ~10GB RAM for 138K users × 9K movies. Using `scipy.csr_matrix` reduces memory footprint to ~120MB. Similarity matrix stored separately as a numpy `.npy` file for fast loading.

**Matrix multiplication for recommendations** — both Item-CF and SVD replace per-user predict loops with matrix multiplication. O(n) vs O(n²) complexity. SVD additionally offers a FAISS IndexFlatIP path for production-scale approximate nearest neighbor retrieval (~5x speedup).

**Objective-aware evaluation** — SVD evaluated with RMSE (its intended metric) alongside Top-N for comparison. Popularity and pure ranking models evaluated with Precision@K, Recall@K, NDCG@K only. Metrics are not compared across groups.

**Movie filtering for Item-CF** — minimum 50 ratings threshold for similarity computation. Movies with fewer ratings produce statistically unreliable similarity scores. ~40% of eligible movies are sparse even after thin-tail removal.

**Graph subset** — Neo4j loaded with top 10K most active users and top 5K most rated movies. Graph traversal benefits most from dense, well-connected nodes. Sparse users contribute little signal and inflate query complexity.

**Confidence weighting for ALS** — explicit ratings converted to confidence signals: `confidence = 1 + alpha * rating`. Lower alpha (20) outperformed higher values — aggressive confidence weighting on explicit ratings amplifies noise rather than signal.

**LightFM genome features** — movie genome tag scores (relevance ≥ 0.6) used as item features. This enables LightFM to leverage content similarity alongside collaborative signals, partially addressing the cold-start problem for items with few ratings.

**TMDB poster fetching** — lazy, per-request via `link.csv` MovieLens→TMDB ID mapping. Stateless app design — no pre-fetching or caching of poster URLs.

**Model persistence** — all models serialized to `models/` via pickle (`train_all_models.py`). Item-CF similarity matrix stored separately as numpy array for faster deserialization. Lazy loading with in-memory caching in the Streamlit app.

---

## Dataset

[MovieLens 20M](https://grouplens.org/datasets/movielens/20m/) — 20 million ratings from 138,493 users on 27,278 movies.

**Preprocessing decisions:**
- Removed movies with fewer than 10 global ratings (thin tail — ~59% of movies, <1% of all ratings)
- All users retained (MovieLens enforces minimum 20 ratings per user)
- Per-user temporal train/test split (80/20 by interaction count)
- Graph subset: top 10K users × top 5K movies

**Key dataset characteristics:**
- 99.46% matrix sparsity
- Mean rating: 3.53 (positive skew)
- Median ratings per user: 54 (mean: 114, heavily right-skewed due to power users)
- Median ratings per movie (after filtering): 89



## Tech Stack

Python 3.11 · scikit-surprise · LightFM · implicit · FAISS · Neo4j · Streamlit · pandas · numpy · scipy · scikit-learn · TMDB API

---

## License

MIT — see [LICENSE](LICENSE)

---

## Author

Built as a portfolio project demonstrating practical ML engineering across algorithm breadth, evaluation rigor, and production-oriented design decisions.
A movie recommendation system exploring 6 algorithms on the MovieLens 20M dataset, with unified evaluation metrics and an interactive Streamlit UI.
