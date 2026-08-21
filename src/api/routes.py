"""Flask API routes for anime recommendations."""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from src.models.hybrid import HybridRecommender
from src.utils.data_loader import DataLoader
from src.utils.preprocessing import DataPreprocessor
import logging

logger = logging.getLogger(__name__)

def create_app():
    """Create and configure Flask application."""
    app = Flask(__name__)
    CORS(app)
    
    # Initialize models and data
    try:
        loader = DataLoader()
        anime_df, ratings_df = loader.load_all()
        
        # Preprocess data
        preprocessor = DataPreprocessor()
        ratings_df_clean = preprocessor.filter_cold_users(ratings_df, min_ratings=3)
        ratings_df_clean = preprocessor.filter_cold_items(ratings_df_clean, min_ratings=2)
        
        user_item_matrix = preprocessor.create_user_item_matrix(ratings_df_clean)
        genre_matrix, genres = preprocessor.create_genre_matrix(anime_df)
        
        # Train models
        recommender = HybridRecommender(cf_weight=0.6, cb_weight=0.4)
        recommender.fit(anime_df, user_item_matrix, genre_matrix)
        
        # Store in app context
        app.loader = loader
        app.recommender = recommender
        app.anime_df = anime_df
        
        logger.info("Models trained successfully")
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
    
    # Routes
    @app.route('/api/health', methods=['GET'])
    def health():
        """Health check endpoint."""
        return jsonify({'status': 'healthy'}), 200
    
    @app.route('/api/recommendations/<int:user_id>', methods=['GET'])
    def get_recommendations(user_id):
        """
        Get anime recommendations for a user.
        
        Query parameters:
        - n: Number of recommendations (default: 10)
        - model: 'hybrid', 'cf', or 'cb' (default: 'hybrid')
        """
        try:
            n = request.args.get('n', 10, type=int)
            model = request.args.get('model', 'hybrid', type=str)
            
            if model == 'hybrid':
                recommendations = app.recommender.recommend(user_id, n)
            else:
                return jsonify({'error': f'Unknown model: {model}'}), 400
            
            # Enrich recommendations with anime details
            result = []
            for anime_id, score in recommendations:
                anime_info = app.loader.get_anime_by_id(anime_id)
                if anime_info:
                    result.append({
                        'anime_id': anime_id,
                        'score': round(score, 3),
                        'name': anime_info.get('name'),
                        'genre': anime_info.get('genre'),
                        'type': anime_info.get('type'),
                        'rating': anime_info.get('rating')
                    })
            
            return jsonify({
                'user_id': user_id,
                'recommendations': result,
                'count': len(result)
            }), 200
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/similar-users/<int:user_id>', methods=['GET'])
    def get_similar_users(user_id):
        """
        Get users similar to the given user.
        
        Query parameters:
        - n: Number of similar users (default: 5)
        """
        try:
            n = request.args.get('n', 5, type=int)
            similar_users = app.recommender.cf_model.get_similar_users(user_id, n)
            
            result = [
                {'user_id': uid, 'similarity': round(sim, 3)}
                for uid, sim in similar_users
            ]
            
            return jsonify({
                'user_id': user_id,
                'similar_users': result,
                'count': len(result)
            }), 200
        except Exception as e:
            logger.error(f"Error finding similar users: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/similar-anime/<int:anime_id>', methods=['GET'])
    def get_similar_anime(anime_id):
        """
        Get anime similar to the given anime.
        
        Query parameters:
        - n: Number of similar anime (default: 5)
        """
        try:
            n = request.args.get('n', 5, type=int)
            similar_anime = app.recommender.cb_model.get_similar_anime(anime_id, n)
            
            result = []
            for sid, similarity in similar_anime:
                anime_info = app.loader.get_anime_by_id(sid)
                if anime_info:
                    result.append({
                        'anime_id': sid,
                        'similarity': round(similarity, 3),
                        'name': anime_info.get('name'),
                        'genre': anime_info.get('genre')
                    })
            
            return jsonify({
                'anime_id': anime_id,
                'similar_anime': result,
                'count': len(result)
            }), 200
        except Exception as e:
            logger.error(f"Error finding similar anime: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/stats', methods=['GET'])
    def get_statistics():
        """
        Get dataset statistics.
        """
        try:
            stats = app.loader.get_statistics()
            return jsonify(stats), 200
        except Exception as e:
            logger.error(f"Error retrieving statistics: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 errors."""
        return jsonify({'error': 'Endpoint not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 errors."""
        return jsonify({'error': 'Internal server error'}), 500
    
    return app
