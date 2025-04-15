import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def create_tfidf_matrix(df):
    tfidf = TfidfVectorizer(stop_words="english")
    # Ensure combined_features doesn't have NaNs after merges/processing
    df["combined_features"] = df["combined_features"].fillna("")
    tfidf_matrix = tfidf.fit_transform(df["combined_features"])
    return tfidf_matrix


def get_recommendations(
    movie_id,  # Changed from title
    df,  # This is movies_combined
    tfidf_matrix,
    movieId_to_index,  # New mapping
    index_to_movieId,  # New mapping
    movieId_to_title,  # New mapping
    top_n=10,
):
    # Check if movie_id exists
    if movie_id not in movieId_to_index:
        # Try to get title for warning, fallback if ID not in movieId_to_title either
        title_for_warning = movieId_to_title.get(movie_id, f"ID {movie_id}")
        print(
            f"Warning: Movie '{title_for_warning}' (ID: {movie_id}) not found in the processed database."
        )
        return pd.Series(dtype="object")

    # Get the index for the TF-IDF matrix
    idx = movieId_to_index[movie_id]

    # Ensure index is within bounds (should be guaranteed by reset_index, but good check)
    if idx >= tfidf_matrix.shape[0]:
        title_for_warning = movieId_to_title.get(movie_id, f"ID {movie_id}")
        print(
            f"Warning: Index {idx} for movie '{title_for_warning}' (ID: {movie_id}) is out of bounds for TF-IDF matrix shape {tfidf_matrix.shape[0]}."
        )
        return pd.Series(dtype="object")

    # Calculate cosine similarity
    cosine_sim = linear_kernel(tfidf_matrix[idx : idx + 1], tfidf_matrix).flatten()

    # Get indices of top N similar movies, excluding the movie itself
    similar_indices = cosine_sim.argsort()[-(top_n + 1) :][::-1]
    # Filter out the original movie's *index*
    similar_indices = [i for i in similar_indices if i != idx][:top_n]

    # Get the recommended movieIds using the index_to_movieId mapping
    # Ensure indices are valid before accessing .loc
    valid_similar_indices = [i for i in similar_indices if i < len(index_to_movieId)]
    if not valid_similar_indices:
        return pd.Series(dtype="object")  # No valid recommendations found

    recommended_movie_ids = index_to_movieId.loc[valid_similar_indices]

    # Get the corresponding slice from the combined dataframe for sorting by rating
    recommended_df = df.iloc[
        valid_similar_indices
    ].copy()  # Get rows using matrix indices

    # Sort by average rating
    recommended_df = recommended_df.sort_values(by="avg_rating", ascending=False)

    # Return the titles of the recommended movies using movieId_to_title mapping on sorted df
    recommended_titles = recommended_df["movieId"].map(movieId_to_title)

    return recommended_titles  # Returns a Series of titles


def get_popular_movies_wr(movies_combined, movieId_to_title, top_n=10):
    """Returns the top N movies based on the pre-calculated weighted rating."""
    popular_movies = movies_combined.sort_values("weighted_rating", ascending=False)
    # Map the movie IDs from the sorted df to titles
    popular_titles = popular_movies.head(top_n)["movieId"].map(movieId_to_title)
    return popular_titles


def get_user_recommendations(
    user_id,
    ratings_df,
    movies_combined,  # This is the df passed to get_recommendations
    tfidf_matrix,
    movieId_to_index,
    index_to_movieId,
    movieId_to_title,
    top_n=10,
    # Added parameter to control fallback logic explicitly if needed,
    # but primary trigger is failure to generate personalized recs
    use_fallback=True,
):
    """
    Generates movie recommendations for a specific user based on movies they rated above their personal average.
    Falls back to globally popular movies (based on Weighted Rating) if no personalized recommendations can be generated.
    Uses movieId as the primary identifier.
    """

    # --- Helper function for Fallback ---
    def _get_fallback_recommendations(user_rated_titles_set):
        print(f"\nFalling back to popular movies for user {user_id}.")
        popular_movies = get_popular_movies_wr(
            movies_combined, movieId_to_title, top_n=top_n * 2
        )  # Get more to filter
        # Filter out movies the user has already rated (using the set for efficiency)
        fallback_recs = popular_movies[~popular_movies.isin(user_rated_titles_set)]
        return fallback_recs.head(top_n)  # Return top N of the filtered list

    # --- Get User Ratings and Rated Titles ---
    user_ratings = ratings_df[ratings_df["userId"] == user_id]

    # Prepare set of titles rated by the user for efficient filtering later
    user_rated_titles_set = set()
    if not user_ratings.empty:
        user_rated_movie_ids = user_ratings["movieId"].unique()
        user_rated_titles_set = {
            movieId_to_title.get(mid)
            for mid in user_rated_movie_ids
            if mid in movieId_to_title and movieId_to_title.get(mid) is not None
        }

    # --- Handle Cold Start: No Ratings ---
    if user_ratings.empty:
        print(f"No ratings found for user {user_id}.")
        if use_fallback:
            # Pass empty set as user hasn't rated anything
            return _get_fallback_recommendations(set())
        else:
            return pd.Series(dtype="object")  # Return empty Series if fallback disabled

    # --- Calculate User's Mean Rating ---
    user_mean_rating = user_ratings["rating"].mean()
    if pd.isna(user_mean_rating):  # Should not happen if ratings exist, but good check
        print(f"Warning: Could not calculate mean rating for user {user_id}.")
        if use_fallback:
            return _get_fallback_recommendations(user_rated_titles_set)
        else:
            return pd.Series(dtype="object")

    # --- Find Highly Rated Movies ---
    high_rated_movie_ids = user_ratings[user_ratings["rating"] > user_mean_rating][
        "movieId"
    ].tolist()

    # --- Handle Cold Start: No Movies Rated Above Average ---
    if not high_rated_movie_ids:
        print(
            f"No movies found rated above user {user_id}'s average rating of {user_mean_rating:.2f}."
        )
        if use_fallback:
            return _get_fallback_recommendations(user_rated_titles_set)
        else:
            return pd.Series(dtype="object")  # Return empty Series if fallback disabled

    # --- Generate Recommendations Based on High Ratings ---
    all_recommended_titles = pd.Series(dtype="object")
    found_similar = False  # Flag to check if get_recommendations returned anything
    for m_id in high_rated_movie_ids:
        if m_id in movieId_to_index:
            recs = get_recommendations(
                m_id,
                movies_combined,
                tfidf_matrix,
                movieId_to_index,
                index_to_movieId,
                movieId_to_title,
                top_n=top_n,  # Get N recs per seed movie
            )
            if not recs.empty:
                found_similar = True
                all_recommended_titles = pd.concat([all_recommended_titles, recs])
        # else: # Optional warning kept commented
        # print(f"Skipping recommendations for rated movieId {m_id} as it's not in the processed movie data.")

    # --- Handle Cold Start: No Similar Movies Found ---
    if not found_similar:  # Check flag
        print(
            f"Could not generate recommendations based on user {user_id}'s highly rated movies (maybe none had similar movies?)."
        )
        if use_fallback:
            return _get_fallback_recommendations(user_rated_titles_set)
        else:
            return pd.Series(dtype="object")

    # --- Aggregate Recommendations and Filter Already Rated ---
    recommendation_counts = all_recommended_titles.value_counts().reset_index()
    recommendation_counts.columns = ["title", "recommendation_score"]

    # Use the pre-calculated set for filtering
    final_recommendations = recommendation_counts[
        ~recommendation_counts["title"].isin(user_rated_titles_set)
    ]

    # --- Handle Cold Start: All Recommendations Were Already Seen ---
    if final_recommendations.empty:
        print(
            f"All potential recommendations for user {user_id} have already been rated."
        )
        if use_fallback:
            return _get_fallback_recommendations(user_rated_titles_set)
        else:
            return pd.Series(dtype="object")

    # --- Sort and Return Top N Personalized Recommendations ---
    final_recommendations = final_recommendations.sort_values(
        by="recommendation_score", ascending=False
    )

    return final_recommendations.head(top_n)["title"]
