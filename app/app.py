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
    ["Mode 1 — Existing User", "Mode 2 — You're Personal Recommendation!"],
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
    st.markdown("*Don't have a MovieLens ID? Tell us your taste and we'll find recommendations for you.*")

    from recommender import (
        GENRE_TAGS, VIBE_TAGS,
        get_popular_movies_by_tags,
        get_content_based_recommendations
    )

    # Initialize session state
    for key, default in [
        ('mode2_stage', 1),
        ('mode2_genres', []),
        ('mode2_vibes', []),
        ('mode2_movies', []),
        ('mode2_recs', []),
        ('mode2_skip_count', 0),
        ('mode2_shown_movies', []),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    stage = st.session_state['mode2_stage']

    # Custom CSS for loading overlay effect
    st.markdown("""
        <style>
        .loading-overlay {
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: rgba(0, 0, 0, 0.7);
            z-index: 9999;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 1.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Progress bar
    st.progress(min(stage / 4, 1.0))
    st.caption(f"Step {min(stage, 4)} of 4")

    # --- Stage 1: Genre ---
    if stage == 1:
        st.subheader("🎭 What kind of movies do you like?")
        st.caption("Select all that apply.")

        selected_genres = []
        genre_list = list(GENRE_TAGS.keys())
        genre_cols = st.columns(5)
        for i, genre in enumerate(genre_list):
            with genre_cols[i % 5]:
                if st.checkbox(genre, key=f"genre_{genre}"):
                    selected_genres.append(genre)

        if st.button("Next →", type="primary", disabled=len(selected_genres) == 0):
            st.session_state['mode2_genres'] = selected_genres
            st.session_state['mode2_stage'] = 2
            st.rerun()

    # --- Stage 2: Vibe ---
    elif stage == 2:
        st.subheader("✨ What vibe are you looking for?")
        st.caption("Select all that apply.")

        selected_vibes = []
        vibe_cols = st.columns(4)
        vibe_list = list(VIBE_TAGS.keys())
        for i, vibe in enumerate(vibe_list):
            with vibe_cols[i % 4]:
                if st.checkbox(vibe, key=f"vibe_{vibe}"):
                    selected_vibes.append(vibe)

        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("← Back"):
                st.session_state['mode2_stage'] = 1
                st.rerun()
        with col2:
            if st.button("Next →", type="primary", disabled=len(selected_vibes) == 0):
                st.session_state['mode2_vibes'] = selected_vibes
                st.session_state['mode2_stage'] = 3
                st.rerun()

    # --- Stage 3: Movie selection ---
    elif stage == 3:
        skip_count = st.session_state['mode2_skip_count']
        shown = st.session_state['mode2_shown_movies']

        # After 3 skips — go straight to recommendations
        if skip_count >= 3:
            st.session_state['mode2_stage'] = 4
            st.rerun()

        st.subheader("🎬 Pick up to 3 movies you've seen and liked")
        if skip_count > 0:
            st.caption(f"Showing new suggestions ({skip_count}/3 skips used)")
        else:
            st.caption("These help us understand your taste better.")

        selected_tag_ids = (
            [GENRE_TAGS[g] for g in st.session_state['mode2_genres']] +
            [VIBE_TAGS[v] for v in st.session_state['mode2_vibes']]
        )

        offset = skip_count * 15
        popular_movies = get_popular_movies_by_tags(selected_tag_ids, n=15, offset=offset)

        # Track shown movies
        new_shown = shown + popular_movies['movieId'].tolist()
        st.session_state['mode2_shown_movies'] = new_shown

        selected_movies = []
        if not popular_movies.empty:
            movie_cols = st.columns(5)
            for i, row in enumerate(popular_movies.itertuples()):
                with movie_cols[i % 5]:
                    details = get_movie_details(row.movieId)
                    if details['poster_url']:
                        st.image(details['poster_url'], use_container_width=True)
                    else:
                        st.image(
                            "https://via.placeholder.com/500x750?text=No+Poster",
                            use_container_width=True
                        )
                    if st.checkbox(row.title, key=f"movie_{row.movieId}"):
                        selected_movies.append(row.movieId)

        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            if st.button("← Back"):
                st.session_state['mode2_stage'] = 2
                st.session_state['mode2_skip_count'] = 0
                st.session_state['mode2_shown_movies'] = []
                st.rerun()
        with col2:
            # Hide "None of these" after 3 skips
            if skip_count < 3:
                if st.button("None of these →"):
                    st.session_state['mode2_skip_count'] = skip_count + 1
                    st.rerun()
        with col3:
            if st.button(
                "Get Recommendations →",
                type="primary",
                disabled=len(selected_movies) == 0
            ):
                if len(selected_movies) > 3:
                    st.warning("Please select up to 3 movies only.")
                else:
                    st.session_state['mode2_movies'] = selected_movies
                    st.session_state['mode2_stage'] = 4
                    st.rerun()

    # --- Stage 4: Loading + Results ---
    elif stage == 4:
        st.subheader("🎬 Your Personalized Recommendations")

        if not st.session_state['mode2_recs']:
            import time

            # Fake loading sequence with dark overlay feel via spinner
            placeholder = st.empty()
            with placeholder.container():
                st.markdown("""
                    <div style='text-align:center; padding: 40px;
                    background: rgba(0,0,0,0.05); border-radius: 12px;'>
                    <h3>🔍 Analysing your taste profile...</h3>
                    </div>
                """, unsafe_allow_html=True)
                time.sleep(1.2)

            with placeholder.container():
                st.markdown("""
                    <div style='text-align:center; padding: 40px;
                    background: rgba(0,0,0,0.05); border-radius: 12px;'>
                    <h3>🎯 Searching our movie database...</h3>
                    </div>
                """, unsafe_allow_html=True)
                time.sleep(1.2)

            with placeholder.container():
                st.markdown("""
                    <div style='text-align:center; padding: 40px;
                    background: rgba(0,0,0,0.05); border-radius: 12px;'>
                    <h3>✨ Ranking recommendations just for you...</h3>
                    </div>
                """, unsafe_allow_html=True)

                liked_movies = st.session_state['mode2_movies']
                shown = st.session_state.get('mode2_shown_movies', [])
                if liked_movies:
                    rec_ids = get_content_based_recommendations(liked_movies, n=10, exclude_movies=shown)
                else:
                    # No movies selected (3 skips) — use tags only
                    selected_tag_ids = (
                        [GENRE_TAGS[g] for g in st.session_state['mode2_genres']] +
                        [VIBE_TAGS[v] for v in st.session_state['mode2_vibes']]
                    )
                    popular = get_popular_movies_by_tags(selected_tag_ids, n=10, offset=45)
                    rec_ids = popular['movieId'].tolist()


        if not rec_ids:
            st.warning("Couldn't generate recommendations. Try different selections.")
        else:
            rec_cols = st.columns(5)
            for i, movie_id in enumerate(rec_ids):
                with rec_cols[i % 5]:
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

        if st.button("Start Over 🔄", type="secondary"):
            for key in [
                'mode2_stage', 'mode2_genres', 'mode2_vibes',
                'mode2_movies', 'mode2_recs',
                'mode2_skip_count', 'mode2_shown_movies'
            ]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()