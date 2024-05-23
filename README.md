# MovieLens 100k Recommender System using Surprise Library

This project demonstrates how to build a recommender system using the MovieLens 100k dataset with the `surprise` library. We use various matrix factorization techniques such as SVD (Singular Value Decomposition), SVD++ (an extension of SVD), and NMF (Non-Negative Matrix Factorization). Additionally, we perform hyperparameter tuning using GridSearchCV to find the best performing model.

## Project Setup

### Prerequisites

Make sure you have the following installed:

- Python 3.8 (or higher)
- `pip` for package management

### Installation

1. Create and activate a virtual environment (optional but recommended):

    ```sh
    python -m venv recommender-env
    source recommender-env/bin/activate  # On Windows use `recommender-env\Scripts\activate`
    ```

2. Install the necessary packages:

    ```sh
    pip install scikit-surprise
    ```

### Dataset

We use the built-in MovieLens 100k dataset provided by the `surprise` library.

## Code Explanation

### Import Libraries

```python
from surprise import Dataset, Reader, SVD, SVDpp, NMF
from surprise.model_selection import cross_validate, GridSearchCV
