"""Collaborative Filtering recommendation model."""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict

class CollaborativeFiltering:
    """
    Collaborative Filtering recommendation model.
    
    Recommends items based on ratings of similar users.
    Uses cosine similarity to find similar users.
    """
    
    def __init__(self, n_neighbors: int = 5, similarity_threshold: float = 0.5):
        """
        Initialize Collaborative Filtering model.
        
        Args:
            n_neighbors: Number of similar users to consider
            similarity_threshold: Minimum similarity score (0-1)
        """
        self.n_neighbors = n_neighbors
        self.similarity_threshold = similarity_threshold
        self.user_item_matrix = None
        self.user_similarity = None
        self.anime_ids = None
    
    def fit(self, user_item_matrix: pd.DataFrame) -> 'CollaborativeFiltering':
        """
        Fit the model using user-item rating matrix.
        
        Args:
            user_item_matrix: DataFrame with users as rows, anime as columns, ratings as values
        """
        self.user_item_matrix = user_item_matrix.copy()
        self.anime_ids = user_item_matrix.columns.tolist()
        
        # Replace 0 with NaN for similarity calculation
        matrix_for_similarity = user_item_matrix.replace(0, np.nan)
        
        # Calculate user-user similarity using cosine similarity
        # Handle NaN by filling with 0 for similarity calculation
        matrix_filled = matrix_for_similarity.fillna(0)
        self.user_similarity = cosine_similarity(matrix_filled)
        self.user_similarity = pd.DataFrame(
            self.user_similarity,
            index=user_item_matrix.index,
            columns=user_item_matrix.index
        )
        
        return self
    
    def recommend(self, user_id: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """
        Get anime recommendations for a user.
        
        Args:
            user_id: Target user ID
            n_recommendations: Number of recommendations to return
        
        Returns:
            List of tuples (anime_id, predicted_rating)
        """
        if self.user_item_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if user_id not in self.user_similarity.index:
            return []
        
        # Get similar users
        similarities = self.user_similarity[user_id].copy()
        similarities = similarities[similarities > self.similarity_threshold]
        similarities = similarities.sort_values(ascending=False)[1:self.n_neighbors + 1]  # Exclude self
        
        if len(similarities) == 0:
            return []
        
        # Get anime rated by similar users but not by target user
        user_rated = self.user_item_matrix.loc[user_id]
        user_rated_anime = user_rated[user_rated > 0].index.tolist()
        
        # Calculate weighted average ratings from similar users
        recommendations = {}
        similar_users = similarities.index.tolist()
        
        for anime_id in self.anime_ids:
            if anime_id not in user_rated_anime:
                ratings = self.user_item_matrix.loc[similar_users, anime_id]
                rated_mask = ratings > 0
                
                if rated_mask.any():
                    # Weighted average by similarity
                    weights = similarities[similar_users]
                    weighted_sum = (ratings * weights).sum()
                    weight_sum = weights[rated_mask].sum()
                    
                    if weight_sum > 0:
                        predicted_rating = weighted_sum / weight_sum
                        recommendations[anime_id] = predicted_rating
        
        # Sort by predicted rating and return top N
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return sorted_recs[:n_recommendations]
    
    def get_similar_users(self, user_id: int, n_users: int = 5) -> List[Tuple[int, float]]:
        """
        Get most similar users to a given user.
        
        Args:
            user_id: Target user ID
            n_users: Number of similar users to return
        
        Returns:
            List of tuples (user_id, similarity_score)
        """
        if self.user_similarity is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        similarities = self.user_similarity[user_id].sort_values(ascending=False)[1:n_users + 1]
        return list(zip(similarities.index, similarities.values))
