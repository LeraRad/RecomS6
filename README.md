# RecomS6 — Movie Recommendation System

A portfolio-grade recommendation system comparing 6 algorithms on the MovieLens 20M dataset. Built to demonstrate practical ML engineering skills across algorithm design, evaluation rigor, and production-oriented decisions.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

RecomS6 implements and evaluates six recommendation approaches — from a simple popularity baseline to a Neo4j graph recommender with genome tag traversal — using a unified evaluation pipeline and a Streamlit UI with TMDB movie posters.

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

---

## LLM-Based Evaluation

In addition to standard offline metrics, recommendations were evaluated 
using a local LLM (Llama 3.1 8B via Ollama) as an independent judge — 
an emerging technique known as **LLM-as-judge evaluation**.

### Methodology

For each of 50 sampled users across all 6 models (300 evaluations total):

1. An algorithmic taste profile is built from the user's rating history — 
   top genres, key genome tags, favourite movies, and rating style
2. The profile and 10 recommendations are sent to Llama 3.1 8B running locally
3. The model scores the recommendations 1-10 and provides structured reasoning, 
   strengths, and weaknesses
4. Results are aggregated and compared across models

All inference runs locally on-device.

### Results

| Model | Avg LLM Score | Quantitative Rank |
|-------|--------------|------------------|
| LightFM | **6.77/10** | 1st (Top-N) |
| Graph | 6.73/10 | 1st (Precision) |
| ALS | 6.55/10 | 3rd |
| Item-CF | 6.54/10 | 2nd (Top-N) |
| Popularity | 6.03/10 | 5th |
| SVD | 6.00/10 | 6th |

LLM evaluation rankings are consistent with quantitative offline metrics — 
LightFM and Graph outperform SVD and Popularity in both evaluation methods. 
This cross-validation strengthens confidence in the findings.

246 of 300 evaluations produced valid scores (54 discarded due to JSON 
parsing failures).

### Limitations

See [Limitations & Future Work](#limitations--future-work) for a detailed 
discussion of evaluation biases and profile depth constraints.

---

## Tech Stack

Python 3.11 · scikit-surprise · LightFM · implicit · FAISS · Neo4j · Streamlit · pandas · numpy · scipy · scikit-learn · TMDB API · Ollama (llama3.1:8b)

---

## Limitations & Future Work

### Evaluation Limitations

**LLM evaluation selection bias**
The LLM-based evaluation was conducted on the top 10,000 most active users 
(required for graph model coverage). Active users are more familiar with 
popular films, which may inflate Popularity baseline scores. A more 
representative evaluation would sample across all activity levels.

**Shallow taste profiles**
User profiles sent to the LLM contain only top genres, top genome tags, 
and favourite movies. This misses negative signals (disliked movies), 
taste nuance within genres, and temporal preference evolution. Richer 
profiles would likely produce more discriminating evaluation scores.

**LLM scoring tendency**
Llama 3.1 8B tends toward middle scores (6-7/10), compressing the 
difference between models. A larger model or more carefully engineered 
prompt might produce wider score distributions.

### Model Limitations

**SVD objective mismatch**
SVD is optimized for rating prediction (RMSE) rather than ranking. 
Strong RMSE (0.80) does not translate to Top-N performance.

**Graph coverage**
Neo4j graph contains only top 10K users and top 5K movies due to 
free-tier constraints. Graph recommendations unavailable for 
less active users.

**Item-CF memory**
Item-CF similarity matrix requires ~950MB on disk. At larger scale 
(100K+ items) this approach becomes infeasible.

**Cold start**
All CF models fail for new users with no rating history. Mode 2 
addresses this with content-based cold start but without 
collaborative signal.

### Future Work

- Richer taste profiles using negative signals and rating distributions
- Online A/B evaluation with real user feedback
- BPR (Bayesian Personalized Ranking) as additional ranking-optimized baseline
- FAISS IVF index for production-scale candidate retrieval
- LLM-generated taste summaries instead of algorithmic profiles
- Hyperparameter optimization via systematic grid search
- Graph model expansion with full dataset using Neo4j AuraDB paid tier

---

## License

MIT — see [LICENSE](LICENSE)

---

## Author

Built as a portfolio project demonstrating practical ML engineering across 
algorithm breadth, evaluation rigor, and production-oriented design decisions. 
Implements 6 recommendation algorithms on MovieLens 20M, evaluated through 
both standard offline metrics and LLM-as-judge evaluation using a locally 
running Llama 3.1 8B model — combining quantitative rigor with qualitative 
reasoning-based assessment.
