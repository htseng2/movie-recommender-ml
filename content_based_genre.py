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
    return agg_ratings


def create_combined_features(movies_df, tags_df, ratings_df=None):
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

    # Add ratings if provided
    if ratings_df is not None:
        agg_ratings = aggregate_ratings(ratings_df)  # Groups by movieId
        movies_combined = pd.merge(
            movies_combined, agg_ratings, on="movieId", how="left"
        )
        # Fill NaNs introduced by left merge if a movie has no ratings
        movies_combined["avg_rating"] = movies_combined["avg_rating"].fillna(0)
        movies_combined["rating_count"] = movies_combined["rating_count"].fillna(0)

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

    # Return the combined dataframe and the new mappings
    return movies_combined, movieId_to_index, index_to_movieId, movieId_to_title


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


def get_user_recommendations(
    user_id,
    ratings_df,
    movies_combined,  # This is the df passed to get_recommendations
    tfidf_matrix,
    movieId_to_index,  # New mapping
    index_to_movieId,  # New mapping
    movieId_to_title,  # New mapping
    top_n=10,
):
    """
    Generates movie recommendations for a specific user based on movies they rated above their personal average.
    Uses movieId as the primary identifier.
    """
    # 1. Get all ratings for the user
    user_ratings = ratings_df[ratings_df["userId"] == user_id]

    if user_ratings.empty:
        print(f"No ratings found for user {user_id}.")
        # Return empty Series for consistency
        return pd.Series(dtype="object")

    # 2. Calculate the user's mean rating
    user_mean_rating = user_ratings["rating"].mean()

    if pd.isna(user_mean_rating):
        print(f"Warning: Could not calculate mean rating for user {user_id}.")
        return pd.Series(dtype="object")

    # 3. Get movieIds rated *above* the user's average rating
    high_rated_movie_ids = user_ratings[user_ratings["rating"] > user_mean_rating][
        "movieId"
    ].tolist()

    if not high_rated_movie_ids:
        print(
            f"No movies found rated above user {user_id}'s average rating of {user_mean_rating:.2f}."
        )
        # Return empty Series for consistency
        return pd.Series(dtype="object")

    # 4. For each highly-rated movie_id, get content-based recommendations
    all_recommended_titles = pd.Series(dtype="object")
    for m_id in high_rated_movie_ids:
        # Check if the movie ID exists in our processed data before getting recs
        if m_id in movieId_to_index:
            recs = get_recommendations(
                m_id,  # Pass movie_id
                movies_combined,
                tfidf_matrix,
                movieId_to_index,
                index_to_movieId,
                movieId_to_title,  # Pass mapping
                top_n=top_n,
            )
            # Concatenate the resulting Series of titles
            all_recommended_titles = pd.concat([all_recommended_titles, recs])
        # else: # Optional: Warn if a rated movie isn't in the combined set
        # print(f"Skipping recommendations for rated movieId {m_id} as it's not in the processed movie data.")

    if all_recommended_titles.empty:
        print(
            f"Could not generate recommendations based on user {user_id}'s highly rated movies (maybe none had similar movies?)."
        )
        # Return empty Series for consistency
        return pd.Series(dtype="object")

    # 5. Aggregate recommendations (titles) and count frequency
    recommendation_counts = all_recommended_titles.value_counts().reset_index()
    recommendation_counts.columns = ["title", "recommendation_score"]

    # 6. Filter out movies the user has already rated
    # Get titles of all movies rated by the user
    user_rated_movie_ids = user_ratings["movieId"].unique()
    # Map these IDs to titles, handling potential missing IDs safely
    user_rated_titles = [
        movieId_to_title.get(mid)
        for mid in user_rated_movie_ids
        if mid in movieId_to_title
    ]

    final_recommendations = recommendation_counts[
        ~recommendation_counts["title"].isin(user_rated_titles)
    ]

    # 7. Sort by recommendation score (frequency) and return top N titles
    final_recommendations = final_recommendations.sort_values(
        by="recommendation_score", ascending=False
    )

    return final_recommendations.head(top_n)["title"]


if __name__ == "__main__":
    # --- Loading Data ---
    print("Loading data...")
    movies = load_movies("/Users/christseng/Downloads/ml-32m/movies.csv")
    tags = load_tags("/Users/christseng/Downloads/ml-32m/tags.csv")
    ratings = load_ratings("/Users/christseng/Downloads/ml-32m/ratings.csv")
    print("Data loaded.")

    # --- Preprocessing and Feature Engineering ---
    print("Creating combined features and mappings...")
    # Now returns movieId based mappings
    movies_combined, movieId_to_index, index_to_movieId, movieId_to_title = (
        create_combined_features(movies, tags, ratings)
    )
    print("Features created.")

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
            print(movie_recs)
        else:
            print(
                f"Movie title '{movie_title_to_find}' not found in the processed data."
            )

    except Exception as e:
        print(f"Error getting recommendations for movie '{movie_title_to_find}': {e}")

    print("-" * 30)

    # --- Get Recommendations for a User ---
    user_id_to_recommend = 3  # Example User ID
    print(f"\nFinding recommendations for user '{user_id_to_recommend}'...")
    try:
        user_recs = get_user_recommendations(
            user_id_to_recommend,
            ratings,
            movies_combined,
            tfidf_matrix,
            movieId_to_index,  # Pass new mapping
            index_to_movieId,  # Pass new mapping
            movieId_to_title,  # Pass new mapping
            top_n=10,
        )
        print(f"\nRecommendations for user '{user_id_to_recommend}':")
        # Check if the result is a Series (recommendations found) or a message/empty Series
        if isinstance(user_recs, pd.Series) and not user_recs.empty:
            print(user_recs)
        elif isinstance(user_recs, pd.Series) and user_recs.empty:
            # Handle case where function returns empty series (e.g., no recs after filtering)
            # The function itself prints messages for most cases (no ratings, no high ratings etc)
            print("No recommendations found for this user after filtering.")
        else:
            print(user_recs)  # Print the message string returned by the function

    except Exception as e:
        print(f"Error getting recommendations for user {user_id_to_recommend}: {e}")
