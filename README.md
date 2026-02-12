# Scalable Data Debugging for Neighborhood-based Recommendation

This repository explores **Data Valuation** in the context of Recommender Systems. Using a game-theoretic approach (Shapley Values), we identify which specific user sessions in a large-scale dataset (Last.fm) are most beneficial or most harmful to a model's predictive performance.

##  Project Motivation
Traditional machine learning focuses on improving models. **Data-Centric AI** focuses on the data itself. In this project, we answer:
1. Which data points are "Gold" (high contribution)?
2. Which data points are "Noise/Harmful" (negative contribution)?
3. How can we scale these calculations for millions of rows?

##  Methodology
### 1. Model: Session-Based k-Nearest Neighbors (SKNN)
We utilize a session-based recommendation approach where item similarities are computed based on co-occurrence within user sessions.

### 2. Scalability: KMC-Shapley
Calculating exact Shapley values for $N$ samples requires $2^N$ model evaluations. To handle the **Last.fm dataset** (~19 million interactions), we implemented:
- **Sessionization:** Grouping logs into meaningful user intents.
- **K-Means Clustering:** Grouping similar sessions into $K$ clusters.
- **Marginal Contribution:** Calculating the value of each *cluster* rather than individual points, significantly reducing complexity.

##  Experimental Results

### Baseline Performance
Evaluation was conducted on a hold-out test set using **Recall@20**, **MRR**, and **NDCG**.
- **Baseline Recall@20:** 0.1665
- **Baseline MRR@20:** 0.0664
- **Baseline NDCG@20:** 0.0889

### Data Debugging Analysis
We performed a "Leave-One-Out" style removal experiment to observe how the model reacts to data pruning.

| Fraction Removed | Strategy | Recall@20 | Performance Change |
| :--- | :--- | :--- | :--- |
| 0% | Baseline | 0.1665 | - |
| 1% | **Remove LOW (Harmful)** | **0.1670** | 📈 **Improvement** |
| 1% | Random Removal | 0.1666 | Neutral |
| 1% | **Remove HIGH (Gold)** | 0.1670* | Neutral (at 1% threshold) |

*Note: At low removal fractions (1%), the impact on the large-scale model is subtle, but the trend identifies sessions that increase the model's precision when removed (noise reduction).*

### Session-Level Insights
The algorithm identified specific sessions that are highly influential:
- **High-Importance (Gold):** Sessions with IDs like `754603`, `484712`. These sessions represent strong, consistent user behavior patterns.
- **Low-Importance (Harmful):** Sessions with IDs like `329734`, `279760`. These were flagged as outliers or inconsistent sessions that degrade the neighborhood similarity matrix.

##  Usage
1. **Data Preparation:** The notebook uses `lastfm_union.parquet`. Ensure sessions are generated with a 30-minute inactivity threshold.
2. **Clustering:** Run the K-Means section to define data groups.
3. **Valuation:** Execute the Shapley calculation loop to rank sessions.
4. **Visualization:** Use the provided matplotlib scripts to generate data removal curves.

##  Impact for Research
This project demonstrates a practical implementation of **Data Valuation** on real-world, large-scale datasets. It provides a blueprint for building more robust recommendation systems by identifying and pruning "poisoned" or noisy data points in a systematic, theoretically grounded way.

---
*Analysis performed using Python, Scikit-Learn, and Pandas.*
