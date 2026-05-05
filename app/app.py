import streamlit as st
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from recommender import (
    get_recommendations, get_user_profile, get_movie_title,
    get_closest_graph_user, get_graph_user_ids, train
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

    # Handle suggested user override
    if 'override_user_id' in st.session_state:
        default_user_id = int(st.session_state['override_user_id'])
        del st.session_state['override_user_id']
    else:
        default_user_id = 1

    col1, col2 = st.columns([1, 2])

    with col1:
        user_id = st.number_input(
            "Enter MovieLens User ID",
            min_value=1,
            max_value=int(train['userId'].max()),
            value=default_user_id,
            step=1
        )

        model_name = st.selectbox(
            "Select Model",
            ["SVD", "Item-CF", "ALS", "LightFM", "Popularity", "Graph"]
        )

        recommend_btn = st.button("Get Recommendations", type="primary")

    with col2:
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
        st.session_state['current_user_id'] = int(user_id)
        st.session_state['current_model'] = model_name

    effective_user_id = st.session_state.get('current_user_id', None)

    if effective_user_id is not None:
        with st.spinner(f"Generating recommendations using {model_name}..."):
            movie_ids = get_recommendations(effective_user_id, model_name, n=10)

        if movie_ids is None:
            graph_user_ids = get_graph_user_ids()
            closest_id = get_closest_graph_user(effective_user_id, graph_user_ids)
            st.warning(
                f"User {effective_user_id} is not in the graph database — only top 10K most active users are included. "
                f"Closest available user by activity: **{closest_id}**"
            )
            if st.button(f"Load recommendations for User {closest_id} instead"):
                st.session_state['override_user_id'] = closest_id
                st.session_state['current_user_id'] = closest_id
                st.rerun()

        elif not movie_ids:
            st.warning("No recommendations generated. Try a different model or user.")

        else:
            st.subheader(f"Top 10 Recommendations — {model_name}")
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