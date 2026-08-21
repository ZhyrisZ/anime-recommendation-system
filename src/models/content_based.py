"""Content-Based recommendation model."""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple

class ContentBased:
    """
    Content-Based recommendation model.
    
    Recommends items similar to those the user has already rated highly.
    Uses item features (genres, type, etc.) to calculate similarity.
    """
    
    def __init__(self):
        self.anime_df = None
        self.user_item_matrix = None
        self.feature_matrix = None
        self.anime_similarity = None
        self.anime_ids = None
    
    def fit(self, anime_df: pd.DataFrame, user_item_matrix: pd.DataFrame, 
            feature_matrix: np.ndarray) -> 'ContentBased':
        """
        Fit the content-based model.
        
        Args:
            anime_df: DataFrame containing anime information
            user_item_matrix: User-item rating matrix
            feature_matrix: Anime feature matrix (genres, type, etc.)
        """
        self.anime_df = anime_df.copy()
        self.user_item_matrix = user_item_matrix.copy()
        self.feature_matrix = feature_matrix
        self.anime_ids = user_item_matrix.columns.tolist()
        
        # Calculate anime-anime similarity based on features
        self.anime_similarity = cosine_similarity(feature_matrix)
        self.anime_similarity = pd.DataFrame(
            self.anime_similarity,
            index=anime_df.index,
            columns=anime_df.index
        )
        
        return self
    
    def recommend(self, user_id: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """
        Get content-based anime recommendations for a user.
        
        Args:
            user_id: Target user ID
            n_recommendations: Number of recommendations to return
        
        Returns:
            List of tuples (anime_id, score)
        """
        if self.user_item_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if user_id not in self.user_item_matrix.index:
            return []
        
        # Get user's rated anime
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_anime = user_ratings[user_ratings > 0]
        
        if len(rated_anime) == 0:
            return []
        
        # Calculate scores for unrated anime based on similarity to rated ones
        recommendations = {}
        rated_indices = rated_anime.index.tolist()
        
        for anime_id in self.anime_ids:
            if anime_id not in rated_indices:
                # Get similarity scores to rated anime
                similarities = self.anime_similarity.loc[anime_id, rated_indices]
                ratings = rated_anime[rated_indices]
                
                # Weighted average: similarity * rating
                score = (similarities * ratings).sum() / similarities.sum()
                recommendations[anime_id] = score
        
        # Sort by score and return top N
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return sorted_recs[:n_recommendations]
    
    def get_similar_anime(self, anime_id: int, n_anime: int = 5) -> List[Tuple[int, float]]:
        """
        Get anime similar to a given anime.
        
        Args:
            anime_id: Target anime ID
            n_anime: Number of similar anime to return
        
        Returns:
            List of tuples (anime_id, similarity_score)
        """
        if self.anime_similarity is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if anime_id not in self.anime_similarity.index:
            return []
        
        similarities = self.anime_similarity[anime_id].sort_values(ascending=False)[1:n_anime + 1]
        return list(zip(similarities.index, similarities.values))
