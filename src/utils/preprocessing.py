"""Data preprocessing utilities."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List

class DataPreprocessor:
    """Preprocess anime and ratings data for recommendation algorithms."""
    
    @staticmethod
    def create_user_item_matrix(ratings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a user-item rating matrix.
        
        Returns a pivot table with users as rows and anime as columns,
        filled with ratings (NaN for unrated items).
        """
        return ratings_df.pivot_table(
            index='user_id',
            columns='anime_id',
            values='rating',
            fill_value=0
        )
    
    @staticmethod
    def filter_cold_users(ratings_df: pd.DataFrame, min_ratings: int = 3) -> pd.DataFrame:
        """
        Filter out users with fewer than min_ratings ratings.
        Helps avoid cold-start problem.
        """
        user_rating_counts = ratings_df.groupby('user_id').size()
        active_users = user_rating_counts[user_rating_counts >= min_ratings].index
        return ratings_df[ratings_df['user_id'].isin(active_users)]
    
    @staticmethod
    def filter_cold_items(ratings_df: pd.DataFrame, min_ratings: int = 2) -> pd.DataFrame:
        """
        Filter out anime with fewer than min_ratings ratings.
        """
        item_rating_counts = ratings_df.groupby('anime_id').size()
        popular_items = item_rating_counts[item_rating_counts >= min_ratings].index
        return ratings_df[ratings_df['anime_id'].isin(popular_items)]
    
    @staticmethod
    def normalize_ratings(ratings_df: pd.DataFrame, scale: Tuple[int, int] = (1, 10)) -> pd.DataFrame:
        """
        Normalize ratings to a specific scale (default 1-10).
        """
        df = ratings_df.copy()
        scaler = MinMaxScaler(feature_range=scale)
        df['rating'] = scaler.fit_transform(df[['rating']])
        return df
    
    @staticmethod
    def create_genre_matrix(anime_df: pd.DataFrame) -> np.ndarray:
        """
        Create a genre one-hot encoding matrix for content-based filtering.
        
        Expected: anime_df has 'genre' column with comma-separated values.
        """
        all_genres = set()
        for genres_str in anime_df['genre'].dropna():
            all_genres.update([g.strip() for g in str(genres_str).split(',')])
        
        all_genres = sorted(list(all_genres))
        genre_matrix = np.zeros((len(anime_df), len(all_genres)))
        
        for idx, genres_str in enumerate(anime_df['genre']):
            if pd.notna(genres_str):
                genres = [g.strip() for g in str(genres_str).split(',')]
                for genre in genres:
                    if genre in all_genres:
                        genre_idx = all_genres.index(genre)
                        genre_matrix[idx, genre_idx] = 1
        
        return genre_matrix, all_genres
    
    @staticmethod
    def remove_already_rated(recommendations: List[int], user_rated: List[int]) -> List[int]:
        """
        Remove anime that user has already rated from recommendations.
        """
        return [anime_id for anime_id in recommendations if anime_id not in user_rated]
