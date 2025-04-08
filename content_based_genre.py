import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def load_movies(filepath):
    # Load movies dataset
    return pd.read_csv(filepath)


def preprocess_genres(genres):
    # Replace pipe symbol with space so that each genre is treated as a separate token
    return genres.replace("|", " ")


def create_tfidf_matrix(df):
    # Preprocess genres column
    df["processed_genres"] = df["genres"].fillna("").apply(preprocess_genres)

    # Use TfidfVectorizer on the processed genres column
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["processed_genres"])
    return tfidf_matrix


def get_recommendations(title, df, tfidf_matrix, top_n=10):
    # Create a series mapping movie titles to the DataFrame index
    indices = pd.Series(df.index, index=df["title"]).drop_duplicates()

    if title not in indices:
        return f"Movie '{title}' not found in the database."

    idx = indices[title]

    # Compute cosine similarity between the movies
    cosine_sim = linear_kernel(tfidf_matrix[idx : idx + 1], tfidf_matrix).flatten()

    # Get indices of the most similar movies (excluding the movie itself)
    similar_indices = cosine_sim.argsort()[-(top_n + 1) :][::-1]
    similar_indices = [i for i in similar_indices if i != idx][:top_n]

    return df["title"].iloc[similar_indices]


if __name__ == "__main__":
    # Use the full path to your movies.csv file
    movies = load_movies("/Users/christseng/Downloads/ml-32m/movies.csv")
    tfidf_matrix = create_tfidf_matrix(movies)
    movie_title = "Toy Story (1995)"
    recommendations = get_recommendations(movie_title, movies, tfidf_matrix)
    print(f"Recommendations based on genres for '{movie_title}':")
    print(recommendations)
