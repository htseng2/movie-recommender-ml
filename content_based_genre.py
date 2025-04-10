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

    return movies_combined


def create_tfidf_matrix(df):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["combined_features"])
    return tfidf_matrix


def get_recommendations(title, df, tfidf_matrix, top_n=10):
    indices = pd.Series(df.index, index=df["title"]).drop_duplicates()
    if title not in indices:
        return f"Movie '{title}' not found in the database."
    idx = indices[title]
    cosine_sim = linear_kernel(tfidf_matrix[idx : idx + 1], tfidf_matrix).flatten()
    similar_indices = cosine_sim.argsort()[-(top_n + 1) :][::-1]
    similar_indices = [i for i in similar_indices if i != idx][:top_n]

    # Optionally, you can re-rank these recommendations using avg_rating
    recommended = df.iloc[similar_indices].copy()
    recommended = recommended.sort_values(by="avg_rating", ascending=False)

    return recommended["title"]


if __name__ == "__main__":
    movies = load_movies("/Users/christseng/Downloads/ml-32m/movies.csv")
    tags = load_tags("/Users/christseng/Downloads/ml-32m/tags.csv")
    ratings = load_ratings("/Users/christseng/Downloads/ml-32m/ratings.csv")
    movies_combined = create_combined_features(movies, tags, ratings)
    tfidf_matrix = create_tfidf_matrix(movies_combined)
    movie_title = "Toy Story (1995)"
    recommendations = get_recommendations(movie_title, movies_combined, tfidf_matrix)
    print(f"Recommendations for '{movie_title}':")
    print(recommendations)
