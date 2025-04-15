import pandas as pd


def load_movies(filepath):
    return pd.read_csv(filepath)


def load_tags(filepath):
    return pd.read_csv(filepath)


def load_ratings(filepath):
    return pd.read_csv(filepath)
