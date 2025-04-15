import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def load_movies(filepath):
    return pd.read_csv(filepath)


def load_tags(filepath):
    return pd.read_csv(filepath)


def load_ratings(filepath):
    return pd.read_csv(filepath)


def preprocess_genres(genres):
    return genres.replace("|", " ")


def aggregate_tags(tags_df):
    # Group by movieId and join all tags for that movie.
    aggregated = (
        tags_df.groupby("movieId")["tag"]
        .apply(lambda x: " ".join(x.astype(str)))
        .reset_index()
    )
    return aggregated


def aggregate_ratings(ratings_df):
    # Calculate average rating and count of ratings per movieId
    agg_ratings = ratings_df.groupby("movieId").agg({"rating": ["mean", "count"]})
    agg_ratings.columns = ["avg_rating", "rating_count"]
    agg_ratings.reset_index(inplace=True)
    # Calculate the global mean rating C
    C = agg_ratings["avg_rating"].mean()
    return agg_ratings, C  # Return C as well


def calculate_weighted_rating(x, m, C):
    """Calculates the weighted rating for a movie."""
    v = x["rating_count"]
    R = x["avg_rating"]
    # Avoid division by zero if m is 0, although m should be > 0
    if v + m == 0:
        return C  # Default to global average if no ratings and m=0
    return (v / (v + m)) * R + (m / (v + m)) * C


def create_combined_features(
    movies_df, tags_df, ratings_df=None, m=500
):  # Added m parameter with default
    # Keep movieId
    movies_processed = movies_df.copy()

    # Preprocess genres
    movies_processed["processed_genres"] = (
        movies_processed["genres"].fillna("").apply(preprocess_genres)
    )

    # Aggregate tags by movieId (already does this)
    agg_tags = aggregate_tags(tags_df)

    # Merge tags with movies using movieId
    movies_combined = pd.merge(movies_processed, agg_tags, on="movieId", how="left")
    movies_combined["tag"] = movies_combined["tag"].fillna("")

    # Combine genres and tags
    movies_combined["combined_features"] = (
        movies_combined["processed_genres"] + " " + movies_combined["tag"]
    )

    # Global Mean Rating C, needed for WR calculation
    C = 0  # Default if no ratings_df
    if ratings_df is not None:
        agg_ratings, C = aggregate_ratings(ratings_df)  # Groups by movieId, returns C
        movies_combined = pd.merge(
            movies_combined, agg_ratings, on="movieId", how="left"
        )
        # Fill NaNs introduced by left merge if a movie has no ratings
        movies_combined["avg_rating"] = movies_combined["avg_rating"].fillna(0)
        movies_combined["rating_count"] = movies_combined["rating_count"].fillna(0)

        # Calculate Weighted Rating (WR)
        # Apply the function row-wise. Make sure m and C are passed.
        movies_combined["weighted_rating"] = movies_combined.apply(
            lambda x: calculate_weighted_rating(x, m, C), axis=1
        )

    else:
        # Add dummy columns if no ratings provided, WR defaults towards C (which is 0 here)
        movies_combined["avg_rating"] = 0.0
        movies_combined["rating_count"] = 0.0
        movies_combined["weighted_rating"] = 0.0

    # Drop duplicates based on movieId *before* creating indices
    movies_combined.drop_duplicates(subset="movieId", keep="first", inplace=True)

    # Reset index after potential drop_duplicates to ensure it's contiguous 0..N-1
    movies_combined.reset_index(drop=True, inplace=True)  # Crucial for TF-IDF alignment

    # Create mappings based on the *final* DataFrame index and movieId
    # Extract numpy arrays to avoid potential index alignment issues
    movie_ids_array = movies_combined["movieId"].to_numpy()
    titles_array = movies_combined["title"].to_numpy()
    df_indices_array = (
        movies_combined.index.to_numpy()
    )  # Should be 0..N-1 after reset_index

    # movieId -> index (row number in movies_combined/TF-IDF matrix)
    movieId_to_index = pd.Series(df_indices_array, index=movie_ids_array)

    # index -> movieId
    index_to_movieId = pd.Series(movie_ids_array, index=df_indices_array)

    # movieId -> title (useful for display)
    movieId_to_title = pd.Series(titles_array, index=movie_ids_array)

    # Return the combined dataframe, the global mean rating, and the new mappings
    # We don't strictly *need* to return C, but it might be useful for debugging/info
    return movies_combined, C, movieId_to_index, index_to_movieId, movieId_to_title


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


if __name__ == "__main__":
    # --- Constants ---
    MIN_RATINGS_THRESHOLD_M = 500  # Threshold for Weighted Rating

    # --- Loading Data ---
    print("Loading data...")
    movies = load_movies("/Users/christseng/Downloads/ml-32m/movies.csv")
    tags = load_tags("/Users/christseng/Downloads/ml-32m/tags.csv")
    ratings = load_ratings("/Users/christseng/Downloads/ml-32m/ratings.csv")
    print("Data loaded.")

    # --- Preprocessing and Feature Engineering ---
    print(
        f"Creating combined features, mappings, and weighted ratings (m={MIN_RATINGS_THRESHOLD_M})..."
    )
    # Now returns movieId based mappings and calculates WR internally
    (
        movies_combined,
        global_mean_rating_C,
        movieId_to_index,
        index_to_movieId,
        movieId_to_title,
    ) = create_combined_features(movies, tags, ratings, m=MIN_RATINGS_THRESHOLD_M)
    print(f"Features created. Global mean rating (C): {global_mean_rating_C:.4f}")
    # Optional: Display top movies by WR
    print("\nTop 10 movies by Weighted Rating:")
    print(get_popular_movies_wr(movies_combined, movieId_to_title, top_n=10))
    print("-" * 30)

    # --- TF-IDF Matrix ---
    # Ensure movies_combined has the reset index for alignment
    print("Creating TF-IDF matrix...")
    tfidf_matrix = create_tfidf_matrix(movies_combined)
    print(f"TF-IDF matrix created with shape: {tfidf_matrix.shape}")

    # --- Get Recommendations for a Movie ---
    movie_title_to_find = "Toy Story (1995)"
    print(f"\nFinding recommendations for movie '{movie_title_to_find}'...")
    # Find the movieId for the title
    try:
        # Find movieIds matching the title
        matching_ids = movieId_to_title[movieId_to_title == movie_title_to_find].index
        if not matching_ids.empty:
            target_movie_id = matching_ids[0]  # Use the first matching ID
            print(f"Found movieId {target_movie_id} for '{movie_title_to_find}'.")

            movie_recs = get_recommendations(
                target_movie_id,  # Pass movie_id
                movies_combined,
                tfidf_matrix,
                movieId_to_index,
                index_to_movieId,
                movieId_to_title,  # Pass mapping
                top_n=10,  # Specify top_n here if desired
            )
            print(
                f"\nRecommendations for movie '{movie_title_to_find}' (ID: {target_movie_id}):"
            )
            # Check if the result is a Series (recommendations found) or empty
            if isinstance(movie_recs, pd.Series) and not movie_recs.empty:
                print(movie_recs.to_string())  # Use to_string for better console output
            else:
                print("No content-based recommendations found for this movie.")
        else:
            print(
                f"Movie title '{movie_title_to_find}' not found in the processed data."
            )

    except Exception as e:
        print(f"Error getting recommendations for movie '{movie_title_to_find}': {e}")

    print("-" * 30)

    # --- Get Recommendations for Users (Including Cold Start Examples) ---
    # Example 1: Existing user (likely to get personalized recs)
    user_id_existing = 3
    # Example 2: User known to have few/no ratings (or use an ID guaranteed not to exist)
    user_id_cold_start_new = -1  # Guaranteed no ratings
    # Example 3: Potentially a user who only rated things below their average
    # (Requires inspecting the data or trying different IDs)
    user_id_cold_start_low_ratings = 999999  # High ID likely not in dataset

    user_ids_to_test = [
        user_id_existing,
        user_id_cold_start_new,
        user_id_cold_start_low_ratings,
    ]

    for user_id in user_ids_to_test:
        print(f"\nFinding recommendations for user '{user_id}'...")
        try:
            user_recs = get_user_recommendations(
                user_id,
                ratings,
                movies_combined,
                tfidf_matrix,
                movieId_to_index,
                index_to_movieId,
                movieId_to_title,
                top_n=10,
            )
            print(f"\nRecommendations for user '{user_id}':")
            # Check if the result is a Series (recommendations found) or empty
            if isinstance(user_recs, pd.Series) and not user_recs.empty:
                # Using .to_string() avoids truncation in pandas display
                print(user_recs.to_string())
            elif isinstance(user_recs, pd.Series) and user_recs.empty:
                # Message should have been printed within the function for empty cases
                # Add a generic one here just in case.
                print("No recommendations found for this user.")
            # else: # Function should always return a Series now
            #    print(user_recs) # Should not happen

        except Exception as e:
            print(f"Error getting recommendations for user {user_id}: {e}")
        print("-" * 30)
