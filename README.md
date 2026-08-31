# Multi-Objective Movie Recommendation with Matrix Factorization and Whale Optimization

This repository contains a movie recommendation system built on the **MovieLens 100K** dataset.

The project combines:

- Matrix Factorization (MF) for preference prediction
- Top-N candidate generation
- Multi-Objective Whale Optimization Algorithm (MOWOA) for recommendation re-ranking
- Genre-based diversity optimization
- Accuracy, diversity, coverage, and exposure evaluation

The main goal is to study the trade-off between **recommendation relevance** and **recommendation diversity**.

---

## Project Overview

The recommendation pipeline is organized as follows:

```text
MovieLens 100K
      |
      v
Data Preprocessing
      |
      v
Per-User Time-Aware Split
      |
      v
Matrix Factorization
      |
      v
Top-N Candidate Generation
      |
      +--------------------+
      |                    |
      v                    v
MF Top-K Baseline     Multi-Objective
                     Whale Optimization
                           |
                           v
                    Optimized Top-K List
      |                    |
      +---------+----------+
                |
                v
      Accuracy + Diversity Evaluation
                |
                v
     Coverage + Exposure Analysis
```

Matrix Factorization is first used to estimate user-item preferences and generate a high-quality candidate set. The Whale Optimization stage then re-ranks those candidates by jointly considering predicted relevance and genre diversity.

---

## Dataset

The notebook uses the **MovieLens 100K** dataset.

The following files are loaded:

```text
u.data
u.user
u.item
u.genre
```

They provide:

- User-item ratings
- Interaction timestamps
- User information
- Movie titles and metadata
- Movie genre indicators

The notebook downloads the dataset automatically using `gdown` and extracts it into the working directory.

---

## Data Preprocessing

The preprocessing pipeline performs the following steps:

1. Load rating, user, movie, and genre files.
2. Convert the 19 binary genre indicators into genre-ID lists for each movie.
3. Merge genre information with user-item interactions.
4. Create an exploded user-item-genre representation for diversity analysis.
5. Save processed rating data for later use.

Two main processed representations are created:

```text
ratings_with_genres
ratings_exploded
```

---

## Time-Aware Train / Validation / Test Split

Interactions are split independently for each user using their timestamps.

The notebook uses:

```text
Test fraction              = 0.10
Validation fraction        = 0.20
Minimum ratings per user   = 5
Minimum test interactions  = 10
Minimum validation items   = 5
Random state               = 42
```

For eligible users:

```text
Oldest interactions  -> Training
Middle interactions  -> Validation
Most recent ratings  -> Test
```

This setup avoids using future user behavior during model training and produces a more realistic recommendation evaluation protocol.

The exploded genre-level dataset is filtered using the same interaction keys so that all dataset representations remain consistent.

---

## Matrix Factorization Baseline

A Matrix Factorization recommender is implemented from scratch using **NumPy** and stochastic gradient descent.

The predicted user-item rating contains:

```text
Global Mean
    +
User Bias
    +
Item Bias
    +
Dot Product of User and Item Latent Factors
```

The current MF configuration is:

| Hyperparameter | Value |
|---|---:|
| Latent factors | 128 |
| Epochs | 50 |
| Learning rate | 0.01 |
| L2 regularization | 0.02 |
| Random seed | 42 |

Validation and test interactions containing users or items that were not observed in the training set are excluded from MF evaluation.

The rating-prediction model is evaluated using:

- RMSE
- MAE

---

## Candidate Generation

After Matrix Factorization training, all unseen training-catalog movies are scored for each evaluation user.

Movies observed by the user in the training or validation sets are excluded.

The current recommendation configuration is:

```text
Candidate set size = 300
Final Top-K size   = 10
Relevant threshold = rating >= 4.0
```

For every evaluation user:

1. MF scores all unseen movies.
2. The highest-scoring 300 items become the candidate set.
3. The 10 highest-scoring candidates form the **MF baseline recommendation list**.
4. The same candidate set is passed to the multi-objective optimizer.

---

## Genre Representation

Each movie is represented by a binary vector with 19 genre dimensions.

Genre information is used to compute recommendation diversity and enforce a minimum genre-coverage constraint.

---

## Intra-List Diversity

Recommendation diversity is measured using pairwise **Jaccard distance** between movie genre vectors.

For every pair of recommended movies, genre similarity is converted into a distance value. The final Intra-List Diversity (ILD) is the average pairwise genre distance across the recommendation list.

A larger ILD indicates a more genre-diverse recommendation list.

---

## Multi-Objective Whale Optimization Algorithm

The notebook implements a **Multi-Objective Whale Optimization Algorithm (MOWOA)** for re-ranking the MF candidate set.

Each whale is represented by a continuous priority vector over the candidate movies.

The `K` positions with the largest values determine the final recommendation list.

### Objective 1: Predicted Relevance

The first objective maximizes the mean Matrix Factorization score of the selected Top-K movies.

```text
f1 = mean predicted MF score of selected items
```

### Objective 2: Recommendation Diversity

The second objective maximizes genre-based Intra-List Diversity.

```text
f2 = mean pairwise Jaccard distance
```

### Genre Constraint

The optimized recommendation list is encouraged to contain at least:

```text
3 distinct genres
```

Constraint violation is evaluated using an epsilon-relaxed formulation that becomes stricter during optimization.

---

## Pareto Archive

MOWOA maintains a Pareto archive of non-dominated solutions.

Solutions are compared using:

1. Constraint violation
2. Predicted relevance (`f1`)
3. Intra-list diversity (`f2`)

A solution dominates another solution when it is no worse in both objectives and strictly better in at least one objective, provided the constraint handling rules are satisfied.

---

## Crowding Distance

Crowding distance is used to maintain diversity among Pareto-optimal solutions.

The archive is sorted independently by each objective. Boundary solutions receive infinite crowding distance, while interior solutions receive normalized distance contributions from neighboring solutions.

When the archive exceeds its capacity, solutions with larger crowding distances are preferred.

---

## Whale Optimization Search

The optimizer uses the main Whale Optimization mechanisms:

- Encircling a selected solution
- Exploration around a random population member
- Spiral movement around a selected solution

The control parameter `a` gradually decreases during optimization, shifting the search from exploration toward exploitation.

The epsilon value used in constraint handling also decreases over time.

---

## MOWOA Hyperparameters

The current configuration is:

| Hyperparameter | Value |
|---|---:|
| Top-K | 10 |
| Candidate size | 300 |
| Minimum distinct genres | 3 |
| Population size | 50 |
| Maximum iterations | 50 |
| Tournament size | 2 |
| Initial epsilon / penalty value | 2.0 |
| Pareto archive capacity | 25 |
| Initial `a` | 2 |
| Whale update probability `p` | 0.5 |
| Seed entry in parameter dictionary | 42 |

After optimization, feasible Pareto solutions are preferred. The final solution is selected using a weighted combination of normalized predicted relevance and diversity:

```text
60% relevance + 40% diversity
```

---

## Recommendation Generation

MOWOA is executed independently for each evaluation user.

For every user:

1. MF generates the Top-300 candidate set.
2. MOWOA optimizes the recommendation priorities.
3. Exactly 10 movies are selected.
4. The output is checked to ensure:
   - Exactly `K` recommendations are returned
   - No duplicate items exist
   - Every recommended item belongs to the MF candidate set
5. Predicted MF scores are saved with the optimized recommendation ranking.

Runtime statistics are also collected, including:

- Total optimization time
- Number of optimized users
- Mean time per user
- Median time per user
- Maximum time per user

---

## Evaluation Metrics

The MF baseline and optimized recommendation lists are evaluated using both relevance-oriented and beyond-accuracy metrics.

### Accuracy Metrics

- **Precision@K**
- **Recall@K**
- **NDCG@K**
- **MRR@K**

A test movie is considered relevant when:

```text
rating >= 4.0
```

### Diversity and Exposure Metrics

- **Intra-List Diversity (ILD)**
- **Catalog Coverage over the training catalog**
- **Catalog Coverage over the complete movie catalog**
- **Gini coefficient**
- **Normalized exposure entropy**
- **Number of unique recommended items**
- **Genre constraint satisfaction rate**

These metrics allow the notebook to study recommendation quality beyond prediction accuracy alone.

---

## Visual Analysis

The notebook contains several visual comparisons between the MF baseline and the optimized recommender.

### Accuracy-Diversity Trade-off

A per-user scatter plot compares:

```text
Mean predicted Top-K rating
            vs
Intra-List Diversity
```

This visualization shows how MOWOA changes the balance between predicted relevance and genre diversity.

### Catalog Coverage vs K

Coverage is measured for recommendation-list sizes from `K = 1` to `K = 20`.

This shows how much of the training catalog is exposed by each recommender as the recommendation depth increases.

### Exposure Distribution

A log-log exposure plot ranks movies according to how frequently they are recommended.

This helps reveal whether recommendation exposure is concentrated on a relatively small set of movies.

### Lorenz Curve

The Lorenz curve compares exposure inequality between MF and the optimized recommender.

A curve closer to the equality line indicates a more balanced distribution of recommendation exposure.

### Genre Exposure

Genre counts are aggregated across recommendation positions to compare how frequently different movie genres are presented by each recommender.

### Qualitative User Comparison

The notebook also displays an example user's recommendation lists side-by-side, including:

- Rank
- Movie ID
- Movie title
- Genres

This provides a qualitative view of how multi-objective re-ranking changes the final recommendations.

---

## Generated Outputs

The notebook writes several intermediate and result files.

Typical directories include:

```text
movielens_data/
results_mf/
results_ea/
```

Generated recommendation CSV files include the MF baseline and optimized Top-K lists.

The visualization section can also generate files such as:

```text
trade_of.png
converge.png
exposure.png
curve.png
genre_exposure.png
```

---

## Requirements

The notebook uses the following main Python packages:

```text
numpy
pandas
matplotlib
gdown
tqdm
jupyter
```

Install them with:

```bash
pip install numpy pandas matplotlib gdown tqdm jupyter
```

---

## Running the Notebook

Clone the repository:

```bash
git clone <https://github.com/hamidmehranfar/movie-recommender-matrix-factorization-whale-optimization.git>
cd <movie-recommender-matrix-factorization-whale-optimization>
```

Install the dependencies:

```bash
pip install numpy pandas matplotlib gdown tqdm jupyter
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
notebook.ipynb
```

Then run the cells sequentially from top to bottom.

---

## Conclusion

This project demonstrates a two-stage recommendation framework in which Matrix Factorization provides relevance-oriented candidate generation and Multi-Objective Whale Optimization re-ranks those candidates to improve recommendation diversity.

The evaluation combines conventional ranking metrics with diversity, catalog coverage, and exposure metrics, making it possible to analyze the practical trade-off between recommendation accuracy and beyond-accuracy objectives.
