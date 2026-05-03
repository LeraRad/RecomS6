import streamlit as st
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from recommender import (
    get_recommendations, get_user_profile, get_movie_title, train
)
from tmdb import get_movie_details

# --- Page config ---
st.set_page_config(
    page_title="RecomS6 — Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 RecomS6 — Movie Recommendation System")
st.markdown("*Comparing 6 recommendation algorithms on MovieLens 20M*")

# --- Mode selector ---
mode = st.sidebar.radio(
    "Select Mode",
    ["Mode 1 — Existing User", "Mode 2 — New User (Coming Soon)"],
)

if mode == "Mode 1 — Existing User":
    st.header("Mode 1 — Recommendations for Existing User")

    col1, col2 = st.columns([1, 2])

    with col1:
        user_id = st.number_input(
            "Enter MovieLens User ID",
            min_value=1,
            max_value=int(train['userId'].max()),
            value=1,
            step=1
        )

        model_name = st.selectbox(
            "Select Model",
            ["SVD", "Item-CF", "ALS", "LightFM", "Popularity", "Graph"]
        )

        recommend_btn = st.button("Get Recommendations", type="primary")

    with col2:
        # User profile
        profile = get_user_profile(int(user_id))
        if profile:
            st.subheader(f"User {user_id} Profile")
            st.write(f"**Total ratings:** {profile['total_ratings']}")
            st.write(f"**Average rating:** {profile['avg_rating']} ⭐")
            st.write(f"**Top genres:** {', '.join(profile['top_genres'])}")
            st.write(f"**Most active year:** {profile['most_active_year']}")
            st.write(f"**Favourite movies:**")
            for movie in profile['top_movies']:
                st.write(f"  • {movie}")
        else:
            st.warning(f"User {user_id} not found in dataset.")

    if recommend_btn:
        if profile is None:
            st.error("User not found. Please enter a valid User ID.")
        else:
            with st.spinner(f"Generating recommendations using {model_name}..."):
                movie_ids = get_recommendations(int(user_id), model_name, n=10)

            if not movie_ids:
                st.warning("No recommendations generated. Try a different model or user.")
            else:
                st.subheader(f"Top 10 Recommendations — {model_name}")
                
                # Display in grid of 5 columns
                cols = st.columns(5)
                for i, movie_id in enumerate(movie_ids):
                    with cols[i % 5]:
                        title = get_movie_title(movie_id)
                        details = get_movie_details(movie_id)
                        
                        if details['poster_url']:
                            st.image(details['poster_url'], use_container_width=True)
                        else:
                            st.image(
                                "https://via.placeholder.com/500x750?text=No+Poster",
                                use_container_width=True
                            )
                        st.caption(f"**{title}**")
                        if details['overview']:
                            with st.expander("Synopsis"):
                                st.write(details['overview'][:200] + "...")

else:
    st.header("Mode 2 — New User Cold Start")
    st.info("Coming soon — select your taste profile and get personalized recommendations.")