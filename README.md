# Scalable Data Debugging for Neighborhood-based Recommendation Systems

##  Project Overview

This project implements scalable data debugging techniques for neighborhood-based (KNN) recommendation systems using the Last.fm music listening dataset. The goal is to identify which training sessions contribute most to recommendation quality using various data valuation methods, particularly **KMC-Shapley** (K-Nearest Neighbors Monte Carlo Shapley).

##  Objectives

1. Build a session-based KNN recommender system
2. Implement multiple data importance scoring methods
3. Compare effectiveness of different data valuation approaches
4. Analyze which training data points contribute most/least to model performance
5. Validate findings through data removal experiments

##  Dataset

**Dataset**: Last.fm Music Listening Dataset
- **Total interactions**: 19,098,852 listening events
- **Unique users**: 992
- **Unique tracks**: 960,402
- **Features**: user_id, timestamp, artist_id, artist_name, track_id, track_name, gender, age, country, registered

### Data Processing Pipeline

1. **Session Construction**
   - Session boundary: 30 minutes of inactivity (1800 seconds)
   - Total sessions created: 908,878

2. **Filtering Criteria**
   - Minimum session length: ≥2 items
   - Minimum item frequency: ≥5 occurrences
   - Sessions after filtering: 792,024

3. **Train/Validation/Test Split** (Temporal)
   - Training: 70% (554,416 sessions)
   - Validation: 15% (118,804 sessions)
   - Test: 15% (118,804 sessions)

4. **Vocabulary Capping**
   - Top N items kept: 50,000 most frequent items
   - Final training sessions: 430,976
   - Final validation sessions: 81,475
   - Final test sessions: 75,900

##  Methodology

### 1. Session-Based KNN Recommender

**Vectorization**:
- MultiLabelBinarizer creates sparse binary vectors
- Training matrix shape: (430,976, 50,000)
- Non-zero entries: 5,731,554

**Recommendation Strategy**:
- Cosine similarity between session vectors
- K=100 nearest neighbors
- Top-K=20 recommendations
- Aggregate scores from similar sessions
- Filter already-seen items

**Feature Engineering**:
Features computed for each session:
- `length`: Number of items in session
- `unique_items`: Count of distinct items
- `avg_popularity`: Mean popularity of items
- `max_popularity`: Maximum popularity
- `popularity_share`: Fraction of high-popularity items (≥90th percentile)
- `first_item_pop`: Popularity of first item
- `last_item_pop`: Popularity of last item

### 2. Data Importance Methods

#### A. Random Baseline
- Assigns random importance scores to each training session
- Used as control to measure improvement over chance

#### B. Heuristic Importance (Session Length)
- Simple proxy: longer sessions might be more informative
- Fast to compute, interpretable

#### C. Leave-One-Out (LOO)
- Remove each training point individually
- Measure performance drop on validation set
- Importance = Baseline performance - Performance without point
- **Evaluated on**: 200 randomly selected sessions from 5,000 sample
- **Validation subset**: 1,000 sessions
- **Base Recall@20**: 0.138

#### D. KMC-Shapley (Primary Method)
**Configuration**:
- Shap train subset: 5,000 sessions (from 50,000 reference set)
- Shap validation: 2,000 sessions
- Monte Carlo iterations (M): 60
- Checkpoints: [200, 400, 800, 1600, 3200, 5000]

**Algorithm**:
1. For each of M iterations:
   - Generate random permutation of training data
   - For each checkpoint size:
     - Train with first t samples
     - Evaluate on validation set
     - Distribute marginal contribution equally among added samples
2. Average Shapley values across all iterations

**Shapley Statistics**:
- Min: 1.367e-05
- Mean: 1.749e-05
- Max: 2.196e-05
- Nonzero values: 5,000 / 5,000 (100%)

##  Results

### Baseline Performance

**Test Set Evaluation** (5,000 test sessions):
- **Recall@20**: 0.1634
- **MRR@20**: 0.0670
- **NDCG@20**: 0.0886

**Alternative Test Evaluation** (2,000 test sessions):
- **Recall@20**: 0.1665
- **MRR@20**: 0.0664
- **NDCG@20**: 0.0890

### Data Removal Experiments

**Removing 1% of Training Data** (Based on KMC-Shapley Scores):

| Method | Action | Recall@20 | Change from Baseline |
|--------|--------|-----------|---------------------|
| Baseline | - | 0.1665 | - |
| KMC-Shapley | Remove LOW importance | 0.1670 | +0.0005 (↑0.3%) |
| KMC-Shapley | Remove HIGH importance | 0.1670 | +0.0005 (↑0.3%) |
| Random | Remove random 1% | 0.1667 ± 0.0002 | +0.0002 (↑0.1%) |

**Key Finding**: Removing either low or high importance data according to KMC-Shapley resulted in marginal performance changes, suggesting the model is relatively robust to small amounts of data removal at 1% level.

### Session Statistics (Training Set)

| Metric | Mean | Std | Min | 25% | 50% | 75% | Max |
|--------|------|-----|-----|-----|-----|-----|-----|
| Length | 16.34 | 37.84 | 2 | 4 | 8 | 17 | 4,860 |
| Unique Items | 13.30 | 21.52 | 1 | 4 | 8 | 15 | 2,108 |
| Avg Popularity | 308.20 | 240.25 | 16 | 143 | 246.97 | 396.20 | 2,840 |
| Max Popularity | 783.74 | 652.16 | 16 | 268 | 595 | 1,123 | 3,203 |
| Popularity Share | 0.336 | 0.304 | 0 | 0 | 0.290 | 0.500 | 1 |
| First Item Pop | 323.86 | 384.53 | 16 | 87 | 179 | 399 | 3,203 |
| Last Item Pop | 296.29 | 355.88 | 16 | 81 | 162 | 358 | 3,203 |

##  Key Insights

1. **Scalability**: KMC-Shapley provides a computationally tractable approximation to exact Shapley values for large-scale recommendation systems

2. **Data Quality**: The method successfully identifies training sessions with varying importance levels

3. **Robustness**: The KNN recommender shows resilience to small amounts (1%) of data removal, whether low or high importance

4. **Session Characteristics**: 
   - Average session length is ~16 items
   - Substantial variability in session lengths (std: 37.84)
   - Strong presence of popular items in sessions

5. **Importance Distribution**: All evaluated sessions received non-zero Shapley values, indicating each contributes to model performance

##  Technologies & Libraries

- **Python 3.x**
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn
- **Visualization**: matplotlib
- **Sparse Matrices**: scipy.sparse
- **Environment**: Google Colab with GPU (T4)

##  Implementation Details

### Core Functions

1. **`compute_session_features(df_sessions, popularity_counter)`**
   - Extracts descriptive features from sessions
   - Computes popularity metrics

2. **`recommend_knn(session_vec, seen_items, X_ref, sessions_ref, k=100, top_k=20)`**
   - Main recommendation function
   - Uses cosine similarity for neighborhood selection
   - Sparse matrix multiplication for efficiency

3. **`evaluate(df_eval)`**
   - Evaluates recommendation quality
   - Returns Recall@20, MRR@20, NDCG@20

4. **`evaluate_with_ref(df_eval, X_ref, sessions_ref, kNN=100, topK=20)`**
   - Flexible evaluation with custom reference set
   - Used in data removal experiments

### Evaluation Metrics

- **Recall@20**: Fraction of test cases where target item appears in top-20
- **MRR@20** (Mean Reciprocal Rank): Average of 1/rank for target items
- **NDCG@20** (Normalized Discounted Cumulative Gain): Position-aware metric with logarithmic discount

## 🚀 How to Use

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib scipy
```

### Running the Notebook

1. Mount Google Drive and load the Last.fm dataset:
```python
from google.colab import drive
drive.mount('/content/drive')
df = pd.read_parquet('/content/drive/MyDrive/lastfm_union.parquet')
```

2. Follow the notebook cells sequentially:
   - Data loading and preprocessing
   - Session construction
   - Filtering and train/test splitting
   - Feature engineering
   - KNN model training
   - Data importance computation
   - Evaluation and visualization

### Key Parameters to Adjust

- `TOP_N_ITEMS`: Vocabulary size (default: 50,000)
- `MAX_TRAIN_FOR_KNN`: Training reference size (default: 50,000)
- `N_SHAP_TRAIN`: Shapley training sample (default: 5,000)
- `N_SHAP_VAL`: Shapley validation sample (default: 2,000)
- `M`: Monte Carlo iterations (default: 60)
- `k`: Number of neighbors (default: 100)
- `top_k`: Recommendation list size (default: 20)

##  Visualizations

The notebook includes several key visualizations:

1. **Session Length Distribution**: Histogram showing distribution of session lengths in training data
2. **Data Debugging Curves**: Performance (Recall@20) vs fraction of data removed, comparing:
   - Removing low-importance data
   - Removing high-importance data
   - Random removal baseline
3. **Shapley Importance Analysis**: Scatter plots exploring relationships between session features and Shapley values

##  Academic Context

This project implements concepts from:
- Data valuation in machine learning
- Shapley values from cooperative game theory
- Session-based recommendation systems
- Neighborhood-based collaborative filtering

**Relevant Research Areas**:
- Data-centric AI
- Explainable AI
- Recommendation systems
- Collaborative filtering

##  Limitations

1. **Computational Cost**: Full Shapley computation requires exponential evaluations; Monte Carlo approximation trades accuracy for feasibility
2. **Sample Size**: Results based on subsets (5,000 training, 2,000 validation for Shapley)
3. **Single Dataset**: Evaluated only on Last.fm music data
4. **Limited Removal Scale**: Experiments conducted at 1% removal level
5. **KNN Baseline**: More sophisticated models (deep learning) might show different patterns

##  Future Work

1. Scale to larger datasets and vocabulary sizes
2. Compare with other data valuation methods (Data Shapley, Influence Functions)
3. Experiment with different removal fractions (5%, 10%, 20%)
4. Apply to other recommendation domains (movies, e-commerce)
5. Integrate with neural recommendation models
6. Explore active learning applications for data collection prioritization
7. Investigate temporal aspects of data importance

##  Citation

If you use this code or methodology in your research, please cite:

```
@software{scalable_data_debugging_recommendation,
  title = {Scalable Data Debugging for Neighborhood-based Recommendation Systems},
  year = {2024},
  note = {Implementation of KMC-Shapley for session-based recommendations}
}
```

##  License

This project is provided for educational and research purposes.

##  Contributing

Contributions are welcome! Areas for improvement:
- Optimization of Shapley computation
- Additional data importance methods
- More comprehensive evaluation metrics
- Extended experiments on diverse datasets

##  Contact

For questions or collaboration opportunities, please open an issue in the repository.

---

**Last Updated**: February 2026  
**Environment**: Google Colab (Python 3.x, GPU: T4)  
**Dataset**: Last.fm (19M+ interactions, 992 users, 960K tracks)
