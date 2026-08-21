"""Data loading utilities for anime and ratings data."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any

class DataLoader:
    """Load and manage anime and user rating data."""
    
    def __init__(self, data_path: str = 'data/'):
        self.data_path = Path(data_path)
        self.anime_df = None
        self.ratings_df = None
    
    def load_anime_data(self, filename: str = 'anime.csv') -> pd.DataFrame:
        """
        Load anime dataset.
        
        Expected columns:
        - anime_id: Unique anime identifier
        - name: Anime title
        - genre: Comma-separated genres
        - type: TV, Movie, OVA, etc.
        - episodes: Number of episodes
        - rating: Average rating (0-10)
        - members: Number of members
        """
        filepath = self.data_path / filename
        self.anime_df = pd.read_csv(filepath)
        self.anime_df['anime_id'] = self.anime_df['anime_id'].astype(int)
        return self.anime_df
    
    def load_ratings_data(self, filename: str = 'ratings.csv') -> pd.DataFrame:
        """
        Load user ratings dataset.
        
        Expected columns:
        - user_id: Unique user identifier
        - anime_id: Anime identifier
        - rating: User rating (1-10, 0 for watched but not rated)
        """
        filepath = self.data_path / filename
        self.ratings_df = pd.read_csv(filepath)
        self.ratings_df['user_id'] = self.ratings_df['user_id'].astype(int)
        self.ratings_df['anime_id'] = self.ratings_df['anime_id'].astype(int)
        self.ratings_df['rating'] = self.ratings_df['rating'].astype(float)
        return self.ratings_df
    
    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load both anime and ratings datasets."""
        self.load_anime_data()
        self.load_ratings_data()
        return self.anime_df, self.ratings_df
    
    def get_anime_by_id(self, anime_id: int) -> Dict[str, Any]:
        """Get anime information by ID."""
        if self.anime_df is None:
            self.load_anime_data()
        
        anime = self.anime_df[self.anime_df['anime_id'] == anime_id]
        if anime.empty:
            return None
        return anime.iloc[0].to_dict()
    
    def get_user_ratings(self, user_id: int) -> pd.DataFrame:
        """Get all ratings for a specific user."""
        if self.ratings_df is None:
            self.load_ratings_data()
        
        return self.ratings_df[self.ratings_df['user_id'] == user_id]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if self.anime_df is None or self.ratings_df is None:
            self.load_all()
        
        return {
            'total_anime': len(self.anime_df),
            'total_users': self.ratings_df['user_id'].nunique(),
            'total_ratings': len(self.ratings_df),
            'avg_rating': self.ratings_df['rating'].mean(),
            'sparsity': 1 - (len(self.ratings_df) / (self.ratings_df['user_id'].nunique() * len(self.anime_df)))
        }
