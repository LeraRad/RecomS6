import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import json
import ollama
import pandas as pd

_movies = None

def _load_movies():
    global _movies
    if _movies is None:
        _movies = pd.read_csv('data/raw/movie.csv')
        _movies['title_clean'] = _movies['title'].apply(_clean_title)
    return _movies


def _clean_title(title: str) -> str:
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


def get_movie_titles(movie_ids: list) -> list:
    """Convert list of movieIds to titles."""
    movies = _load_movies()
    titles = []
    for mid in movie_ids:
        match = movies[movies['movieId'] == mid]
        if not match.empty:
            titles.append(match.iloc[0]['title_clean'])
        else:
            titles.append(f"Movie {mid}")
    return titles


def evaluate_recommendations(
    profile_text: str,
    movie_ids: list,
    model_name: str = "llama3.1:8b"
) -> dict:
    """
    Ask Ollama to evaluate how well recommendations match a taste profile.
    
    Returns dict with:
        - score: int 1-10
        - reasoning: str
        - strengths: list
        - weaknesses: list
    """
    titles = get_movie_titles(movie_ids)
    titles_text = "\n".join([f"{i+1}. {t}" for i, t in enumerate(titles)])

    prompt = f"""You are an expert movie recommendation evaluator.

Given a user's taste profile and a list of recommended movies, evaluate how well the recommendations match the user's taste.

{profile_text}

Recommended movies:
{titles_text}

Evaluate these recommendations and respond ONLY with a JSON object in this exact format, no other text:
{{
    "score": <integer 1-10>,
    "reasoning": "<2-3 sentence explanation>",
    "strengths": ["<strength 1>", "<strength 2>"],
    "weaknesses": ["<weakness 1>", "<weakness 2>"]
}}

Scoring guide:
1-3: Poor match, recommendations ignore user's taste completely
4-5: Weak match, some relevant recommendations but mostly off
6-7: Decent match, several recommendations align with taste
8-9: Strong match, most recommendations fit the user's profile well
10: Excellent match, all recommendations are highly relevant"""

    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1}  # low temperature for consistent scoring
        )

        content = response['message']['content'].strip()

        # Clean potential markdown code blocks
        if content.startswith('```'):
            content = content.split('```')[1]
            if content.startswith('json'):
                content = content[4:]

        result = json.loads(content)

        # Validate score range
        result['score'] = max(1, min(10, int(result['score'])))
        return result

    except json.JSONDecodeError:
        return {
            'score': None,
            'reasoning': 'Failed to parse LLM response',
            'strengths': [],
            'weaknesses': []
        }
    except Exception as e:
        return {
            'score': None,
            'reasoning': f'Error: {str(e)}',
            'strengths': [],
            'weaknesses': []
        }