"""Unit tests for recommendation models."""

import unittest
import numpy as np
import pandas as pd
from src.models.collaborative_filtering import CollaborativeFiltering
from src.models.content_based import ContentBased
from src.models.hybrid import HybridRecommender
from src.utils.preprocessing import DataPreprocessor

class TestCollaborativeFiltering(unittest.TestCase):
    """Tests for collaborative filtering model."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample user-item matrix
        data = {
            1: [5, 4, 0, 2],
            2: [3, 0, 4, 1],
            3: [4, 3, 5, 0],
            4: [0, 2, 3, 4]
        }
        self.user_item_matrix = pd.DataFrame(data).T
        self.user_item_matrix.columns = [101, 102, 103, 104]
    
    def test_fit(self):
        """Test model fitting."""
        model = CollaborativeFiltering()
        model.fit(self.user_item_matrix)
        
        self.assertIsNotNone(model.user_similarity)
        self.assertEqual(len(model.user_similarity), len(self.user_item_matrix))
    
    def test_recommend(self):
        """Test recommendation generation."""
        model = CollaborativeFiltering()
        model.fit(self.user_item_matrix)
        
        recommendations = model.recommend(1, n_recommendations=2)
        
        self.assertIsInstance(recommendations, list)
        self.assertLessEqual(len(recommendations), 2)
    
    def test_similar_users(self):
        """Test similar user finding."""
        model = CollaborativeFiltering()
        model.fit(self.user_item_matrix)
        
        similar = model.get_similar_users(1, n_users=2)
        
        self.assertIsInstance(similar, list)
        self.assertLessEqual(len(similar), 2)

class TestContentBased(unittest.TestCase):
    """Tests for content-based model."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        self.anime_df = pd.DataFrame({
            'anime_id': [101, 102, 103, 104],
            'name': ['Anime A', 'Anime B', 'Anime C', 'Anime D'],
            'genre': ['Action,Adventure', 'Comedy,School', 'Action,Adventure', 'Drama']
        })
        
        data = {
            1: [5, 4, 0, 2],
            2: [3, 0, 4, 1],
            3: [4, 3, 5, 0]
        }
        self.user_item_matrix = pd.DataFrame(data).T
        self.user_item_matrix.columns = [101, 102, 103, 104]
        
        # Create simple feature matrix
        self.feature_matrix = np.array([
            [1, 1, 0, 0],  # Action, Adventure
            [0, 0, 1, 1],  # Comedy, School
            [1, 1, 0, 0],  # Action, Adventure
            [0, 0, 0, 1]   # Drama
        ])
    
    def test_fit(self):
        """Test model fitting."""
        model = ContentBased()
        model.fit(self.anime_df, self.user_item_matrix, self.feature_matrix)
        
        self.assertIsNotNone(model.anime_similarity)
    
    def test_recommend(self):
        """Test recommendation generation."""
        model = ContentBased()
        model.fit(self.anime_df, self.user_item_matrix, self.feature_matrix)
        
        recommendations = model.recommend(1, n_recommendations=2)
        
        self.assertIsInstance(recommendations, list)

class TestHybridRecommender(unittest.TestCase):
    """Tests for hybrid recommender."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.anime_df = pd.DataFrame({
            'anime_id': [101, 102, 103, 104],
            'name': ['Anime A', 'Anime B', 'Anime C', 'Anime D'],
            'genre': ['Action,Adventure', 'Comedy,School', 'Action,Adventure', 'Drama']
        })
        
        data = {
            1: [5, 4, 0, 2],
            2: [3, 0, 4, 1],
            3: [4, 3, 5, 0]
        }
        self.user_item_matrix = pd.DataFrame(data).T
        self.user_item_matrix.columns = [101, 102, 103, 104]
        
        self.feature_matrix = np.array([
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [0, 0, 0, 1]
        ])
    
    def test_recommend(self):
        """Test hybrid recommendations."""
        model = HybridRecommender()
        model.fit(self.anime_df, self.user_item_matrix, self.feature_matrix)
        
        recommendations = model.recommend(1, n_recommendations=2)
        
        self.assertIsInstance(recommendations, list)
        self.assertGreaterEqual(len(recommendations), 0)

if __name__ == '__main__':
    unittest.main()
