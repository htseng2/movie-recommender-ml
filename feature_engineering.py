import pandas as pd


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
