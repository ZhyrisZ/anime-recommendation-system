# Anime Recommendation System

An intelligent recommendation system for anime and manga using collaborative filtering, content-based filtering, and hybrid algorithms.

## Features

- **Collaborative Filtering**: Recommends anime based on similar users' ratings
- **Content-Based Filtering**: Recommends anime similar to ones the user has rated highly
- **Hybrid Model**: Combines both approaches for robust recommendations
- **RESTful API**: Easy-to-use endpoints for getting recommendations
- **Scalable Architecture**: Designed for easy extension and optimization

## Project Structure

```
anime-recommendation-system/
├── src/
│   ├── models/              # Recommendation algorithms
│   │   ├── collaborative_filtering.py
│   │   ├── content_based.py
│   │   └── hybrid.py
│   ├── utils/               # Data processing utilities
│   │   ├── data_loader.py
│   │   └── preprocessing.py
│   └── api/                 # Flask API
│       └── routes.py
├── data/                    # Anime and ratings datasets
├── tests/                   # Unit tests
├── app.py                   # Application entry point
├── config.py                # Configuration settings
└── requirements.txt         # Python dependencies
```

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/ZhyrisZ/anime-recommendation-system.git
cd anime-recommendation-system
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. **Prepare data**:
   - Place `anime.csv` and `ratings.csv` in the `data/` directory
   - Expected columns in `anime.csv`: anime_id, name, genre, type, episodes, rating, members
   - Expected columns in `ratings.csv`: user_id, anime_id, rating

6. **Run the application**:
```bash
python app.py
```

The API will be available at `http://localhost:5000`

## API Endpoints

### Get Recommendations

```bash
GET /api/recommendations/<user_id>?n=10&model=hybrid
```

**Parameters**:
- `user_id` (int): Target user ID
- `n` (int, optional): Number of recommendations (default: 10)
- `model` (string, optional): Recommendation model - 'hybrid', 'cf', 'cb' (default: 'hybrid')

**Response**:
```json
{
  "user_id": 123,
  "recommendations": [
    {
      "anime_id": 5,
      "score": 0.95,
      "name": "Cowboy Bebop",
      "genre": "Action,Adventure,Sci-Fi",
      "type": "TV",
      "rating": 8.76
    }
  ],
  "count": 10
}
```

### Get Similar Users

```bash
GET /api/similar-users/<user_id>?n=5
```

Find users with similar taste to the given user.

### Get Similar Anime

```bash
GET /api/similar-anime/<anime_id>?n=5
```

Find anime similar to the given anime based on features.

### Get Statistics

```bash
GET /api/stats
```

Get dataset statistics.

### Health Check

```bash
GET /api/health
```

Check if the API is running.

## Recommendation Algorithms

### Collaborative Filtering

Based on the assumption that users who have rated items similarly in the past will rate items similarly in the future.

- Uses cosine similarity to find similar users
- Predicts ratings based on weighted average of similar users' ratings
- Handles sparse data well with cold-start problem mitigation

**Advantages**:
- Works well with large user bases
- Captures complex patterns in user preferences
- No need for item metadata

**Disadvantages**:
- Cold-start problem for new users and items
- Sparsity in rating matrices

### Content-Based Filtering

Recommends items similar to those the user has liked, based on item features.

- Uses genre and anime attributes as features
- Calculates anime-anime similarity based on feature overlap
- Recommends items with high similarity to user's rated items

**Advantages**:
- No cold-start problem for new items
- Transparent and explainable recommendations
- Works with new users (if they have rated at least one item)

**Disadvantages**:
- Limited by feature engineering
- May not capture subtle patterns
- Tends to recommend similar items

### Hybrid Model

Combines both collaborative filtering (60% weight) and content-based filtering (40% weight) to leverage strengths of both approaches.

**Process**:
1. Generate recommendations from both models
2. Normalize scores to 0-1 range
3. Combine scores using weighted averaging
4. Return top N recommendations

## Configuration

Edit `config.py` to adjust:

- `MIN_RATINGS`: Minimum ratings required for a user to get recommendations
- `TOP_N_RECOMMENDATIONS`: Default number of recommendations
- `COLLABORATIVE_N_NEIGHBORS`: Number of similar users to consider
- `CONTENT_WEIGHT` / `COLLABORATIVE_WEIGHT`: Weights for hybrid model

## Testing

Run unit tests:

```bash
python -m pytest tests/ -v
```

Or:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

## Performance Optimization

- **Caching**: Implement caching for frequently requested recommendations
- **Matrix Factorization**: Replace cosine similarity with SVD for better CF
- **Batch Processing**: Process recommendations in batches for efficiency
- **Incremental Learning**: Update models with new ratings incrementally
- **GPU Acceleration**: Use libraries like CuPy for large-scale operations

## Data Requirements

### anime.csv
```
anime_id,name,genre,type,episodes,rating,members
1,Cowboy Bebop,Action;Adventure;Sci-Fi,TV,26,8.76,1200000
```

### ratings.csv
```
user_id,anime_id,rating
1,5,9
1,10,8
```

## Future Enhancements

- [ ] Matrix Factorization (SVD, NMF)
- [ ] Deep Learning models (Neural Collaborative Filtering)
- [ ] Context-aware recommendations (time, mood, etc.)
- [ ] Serendipitous recommendations
- [ ] A/B testing framework
- [ ] Recommendation explanation system
- [ ] Multi-language support for manga
- [ ] Real-time streaming recommendations

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- [Collaborative Filtering](https://en.wikipedia.org/wiki/Collaborative_filtering)
- [Content-based Filtering](https://en.wikipedia.org/wiki/Recommender_system#Content-based_filtering)
- [Hybrid Recommender Systems](https://en.wikipedia.org/wiki/Recommender_system#Hybrid_recommender_system)
- [Scikit-learn Documentation](https://scikit-learn.org/)

## Contact

For questions or suggestions, please open an issue or contact the maintainer.
