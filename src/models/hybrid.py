"""Hybrid recommendation model combining multiple algorithms."""

from typing import List, Tuple, Dict
from .collaborative_filtering import CollaborativeFiltering
from .content_based import ContentBased

class HybridRecommender:
    """
    Hybrid recommendation model.
    
    Combines Collaborative Filtering and Content-Based filtering
    using weighted averaging for robust recommendations.
    """
    
    def __init__(self, cf_weight: float = 0.6, cb_weight: float = 0.4):
        """
        Initialize Hybrid Recommender.
        
        Args:
            cf_weight: Weight for collaborative filtering (0-1)
            cb_weight: Weight for content-based filtering (0-1)
        """
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        
        self.cf_model = CollaborativeFiltering()
        self.cb_model = ContentBased()
    
    def fit(self, anime_df, user_item_matrix, feature_matrix) -> 'HybridRecommender':
        """
        Fit both collaborative and content-based models.
        
        Args:
            anime_df: DataFrame with anime information
            user_item_matrix: User-item rating matrix
            feature_matrix: Anime feature matrix
        """
        self.cf_model.fit(user_item_matrix)
        self.cb_model.fit(anime_df, user_item_matrix, feature_matrix)
        return self
    
    def recommend(self, user_id: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """
        Get hybrid recommendations combining both models.
        
        Args:
            user_id: Target user ID
            n_recommendations: Number of recommendations to return
        
        Returns:
            List of tuples (anime_id, combined_score)
        """
        # Get recommendations from both models
        cf_recs = self.cf_model.recommend(user_id, n_recommendations * 2)
        cb_recs = self.cb_model.recommend(user_id, n_recommendations * 2)
        
        # Normalize scores to 0-1 range
        cf_dict = self._normalize_scores({anime_id: score for anime_id, score in cf_recs})
        cb_dict = self._normalize_scores({anime_id: score for anime_id, score in cb_recs})
        
        # Combine scores
        combined_scores = {}
        all_anime = set(cf_dict.keys()) | set(cb_dict.keys())
        
        for anime_id in all_anime:
            cf_score = cf_dict.get(anime_id, 0) * self.cf_weight
            cb_score = cb_dict.get(anime_id, 0) * self.cb_weight
            combined_scores[anime_id] = cf_score + cb_score
        
        # Sort and return top N
        sorted_recs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_recs[:n_recommendations]
    
    @staticmethod
    def _normalize_scores(scores_dict: Dict[int, float]) -> Dict[int, float]:
        """
        Normalize scores to 0-1 range.
        """
        if not scores_dict:
            return {}
        
        min_score = min(scores_dict.values())
        max_score = max(scores_dict.values())
        
        if max_score == min_score:
            return {k: 0.5 for k in scores_dict}
        
        return {
            k: (v - min_score) / (max_score - min_score)
            for k, v in scores_dict.items()
        }
    
    def get_model_breakdown(self, user_id: int, n_recommendations: int = 10) -> Dict:
        """
        Get recommendations with individual model scores for analysis.
        
        Returns:
            Dict with combined recommendations and individual model scores
        """
        cf_recs = dict(self.cf_model.recommend(user_id, n_recommendations * 2))
        cb_recs = dict(self.cb_model.recommend(user_id, n_recommendations * 2))
        
        return {
            'collaborative_filtering': cf_recs,
            'content_based': cb_recs,
            'hybrid': dict(self.recommend(user_id, n_recommendations))
        }
