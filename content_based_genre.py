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
    # Preprocess genres
    movies_df["processed_genres"] = (
        movies_df["genres"].fillna("").apply(preprocess_genres)
    )

    # Aggregate tags by movieId
    agg_tags = aggregate_tags(tags_df)

    # Merge tags with movies (left join to retain all movies)
    movies_combined = pd.merge(movies_df, agg_tags, on="movieId", how="left")
    movies_combined["tag"] = movies_combined["tag"].fillna("")

    # Combine genres and tags into one string
    movies_combined["combined_features"] = (
        movies_combined["processed_genres"] + " " + movies_combined["tag"]
    )

    if ratings_df is not None:
        agg_ratings = aggregate_ratings(ratings_df)
        movies_combined = pd.merge(
            movies_combined, agg_ratings, on="movieId", how="left"
        )
        movies_combined["avg_rating"] = movies_combined["avg_rating"].fillna(0)
        movies_combined["rating_count"] = movies_combined["rating_count"].fillna(0)

    # Create indices mapping title to index and index to title for quick lookups
    movies_combined["original_index"] = movies_combined.index
    title_to_index = pd.Series(
        movies_combined.index, index=movies_combined["title"]
    ).drop_duplicates()
    index_to_title = pd.Series(
        movies_combined["title"], index=movies_combined.index
    ).drop_duplicates()

    return movies_combined, title_to_index, index_to_title


def create_tfidf_matrix(df):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["combined_features"])
    return tfidf_matrix


def get_recommendations(
    title, df, tfidf_matrix, title_to_index, index_to_title, top_n=10
):
    if title not in title_to_index:
        print(f"Warning: Movie '{title}' not found in the database.")
        return pd.Series(dtype="object")  # Return empty series if title not found
    idx = title_to_index[title]
    # Ensure index is within bounds
    if idx >= tfidf_matrix.shape[0]:
        print(f"Warning: Index for movie '{title}' is out of bounds.")
        return pd.Series(dtype="object")

    # Calculate cosine similarity
    cosine_sim = linear_kernel(tfidf_matrix[idx : idx + 1], tfidf_matrix).flatten()
    # Get indices of top N similar movies, excluding the movie itself
    similar_indices = cosine_sim.argsort()[-(top_n + 1) :][::-1]
    similar_indices = [i for i in similar_indices if i != idx][:top_n]

    # Get titles and potentially re-rank by average rating
    recommended = df.iloc[similar_indices].copy()
    recommended = recommended.sort_values(by="avg_rating", ascending=False)

    return recommended["title"]


def get_user_recommendations(
    user_id,
    ratings_df,
    movies_combined,
    tfidf_matrix,
    title_to_index,
    index_to_title,
    top_n=10,
    rating_threshold=4.0,
):
    """
    Generates movie recommendations for a specific user based on their highly-rated movies.
    """
    # 1. Get movies rated highly by the user
    user_ratings = ratings_df[ratings_df["userId"] == user_id]
    high_rated_movies = user_ratings[user_ratings["rating"] >= rating_threshold]

    if high_rated_movies.empty:
        return f"No movies found with rating >= {rating_threshold} for user {user_id}."

    # Map movieIds to titles using the movies_combined dataframe
    high_rated_movie_titles = pd.merge(
        high_rated_movies,
        movies_combined[["movieId", "title"]],
        on="movieId",
        how="inner",
    )["title"]

    # 2. For each highly-rated movie, get content-based recommendations
    all_recommendations = pd.Series(dtype="object")
    for title in high_rated_movie_titles:
        # Pass the necessary indices mappings to get_recommendations
        recs = get_recommendations(
            title,
            movies_combined,
            tfidf_matrix,
            title_to_index,
            index_to_title,
            top_n=top_n,
        )
        all_recommendations = pd.concat([all_recommendations, recs])

    # 3. Aggregate recommendations and remove duplicates
    all_recommendations = all_recommendations.value_counts().reset_index()
    all_recommendations.columns = [
        "title",
        "recommendation_score",
    ]  # Score based on frequency

    # 4. Filter out movies the user has already rated
    user_rated_titles = pd.merge(
        user_ratings, movies_combined[["movieId", "title"]], on="movieId", how="inner"
    )["title"].unique()

    final_recommendations = all_recommendations[
        ~all_recommendations["title"].isin(user_rated_titles)
    ]

    # 5. Sort by recommendation score (frequency) and return top N
    final_recommendations = final_recommendations.sort_values(
        by="recommendation_score", ascending=False
    )

    return final_recommendations.head(top_n)["title"]


if __name__ == "__main__":
    # --- Loading Data ---
    movies = load_movies("/Users/christseng/Downloads/ml-32m/movies.csv")
    tags = load_tags("/Users/christseng/Downloads/ml-32m/tags.csv")
    ratings = load_ratings("/Users/christseng/Downloads/ml-32m/ratings.csv")

    # --- Preprocessing and Feature Engineering ---
    # Pass ratings to include avg_rating and rating_count
    movies_combined, title_to_index, index_to_title = create_combined_features(
        movies, tags, ratings
    )

    # --- TF-IDF Matrix ---
    tfidf_matrix = create_tfidf_matrix(movies_combined)

    # --- Get Recommendations for a Movie ---
    movie_title = "Toy Story (1995)"
    # Pass the indices to the function
    movie_recs = get_recommendations(
        movie_title, movies_combined, tfidf_matrix, title_to_index, index_to_title
    )
    print(f"Recommendations for movie '{movie_title}':")
    print(movie_recs)
    print("-" * 30)

    # --- Get Recommendations for a User ---
    user_id_to_recommend = 1  # Example User ID
    user_recs = get_user_recommendations(
        user_id_to_recommend,
        ratings,
        movies_combined,
        tfidf_matrix,
        title_to_index,
        index_to_title,
        top_n=10,
        rating_threshold=4.0,
    )
    print(f"\nRecommendations for user '{user_id_to_recommend}':")
    print(user_recs)
