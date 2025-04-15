import pandas as pd
import data_loader
import feature_engineering
import recommender

# --- Constants ---
MIN_RATINGS_THRESHOLD_M = 500  # Threshold for Weighted Rating
# TODO: Update these paths or use command-line arguments/config files
MOVIES_PATH = "/Users/christseng/Downloads/ml-32m/movies.csv"
TAGS_PATH = "/Users/christseng/Downloads/ml-32m/tags.csv"
RATINGS_PATH = "/Users/christseng/Downloads/ml-32m/ratings.csv"


def run_pipeline():
    # --- Loading Data ---
    print("Loading data...")
    movies = data_loader.load_movies(MOVIES_PATH)
    tags = data_loader.load_tags(TAGS_PATH)
    ratings = data_loader.load_ratings(RATINGS_PATH)
    print("Data loaded.")

    # --- Preprocessing and Feature Engineering ---
    print(
        f"Creating combined features, mappings, and weighted ratings (m={MIN_RATINGS_THRESHOLD_M})..."
    )
    (
        movies_combined,
        global_mean_rating_C,
        movieId_to_index,
        index_to_movieId,
        movieId_to_title,
    ) = feature_engineering.create_combined_features(
        movies, tags, ratings, m=MIN_RATINGS_THRESHOLD_M
    )
    print(f"Features created. Global mean rating (C): {global_mean_rating_C:.4f}")

    # Optional: Display top movies by WR
    print("\nTop 10 movies by Weighted Rating:")
    print(
        recommender.get_popular_movies_wr(movies_combined, movieId_to_title, top_n=10)
    )
    print("-" * 30)

    # --- TF-IDF Matrix ---
    # Ensure movies_combined has the reset index for alignment
    print("Creating TF-IDF matrix...")
    tfidf_matrix = recommender.create_tfidf_matrix(movies_combined)
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

            movie_recs = recommender.get_recommendations(
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
            user_recs = recommender.get_user_recommendations(
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


if __name__ == "__main__":
    run_pipeline()
