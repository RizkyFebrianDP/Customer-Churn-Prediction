# Telco Customer Churn Prediction

## Project Description
This project aims to predict customer churn for a telecommunications company. By analyzing customer data, we build and evaluate various machine learning models to identify the key factors driving churn and predict which customers are most likely to leave. The project follows a two-step approach: first, a broad search for the best-performing model type using Randomized Search, followed by a more focused hyperparameter tuning of the selected model using Grid Search.

## Dataset
The project uses the "Telco-Customer-Churn.csv" dataset, which can be found in the `src/data/` directory. This dataset contains customer information such as demographics, account details, and services they have signed up for. The target variable is 'Churn', indicating whether a customer has left the company.

## Installation
To set up the environment and run the project, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    The required Python libraries are listed in `requirements.txt`. You can install them using pip.

    *Note: The `requirements.txt` file is encoded in UTF-16. If you encounter issues, you may need to convert it to UTF-8 or install the packages individually.*

    ```bash
    pip install -r requirements.txt
    ```

## Workflow
The modeling process is divided into two main scripts, located in the `src/models/` directory.

### Step 1: Finding the Best Model with Randomized Search
The first step is to identify the most suitable machine learning model for our churn prediction task. This is done using the `random_search_models.py` script.

This script performs the following actions:
-   Loads and preprocesses the Telco Customer Churn data.
-   Handles class imbalance using the SMOTE (Synthetic Minority Over-sampling Technique).
-   Uses `RandomizedSearchCV` to evaluate several different classifiers, including:
    -   Logistic Regression
    -   Gradient Boosting
    -   Random Forest
    -   Support Vector Classifier (SVC)
    -   LightGBM
    -   CatBoost
-   It then prints out the best-performing model, its optimal hyperparameters, and its performance metrics.

To run this script:
```bash
python src/models/random_search_models.py
```

### Step 2: Fine-tuning the Best Model with Grid Search
Once the best model type has been identified (e.g., RandomForestClassifier), the next step is to perform a more exhaustive search for the best hyperparameters for that specific model. This is done using the `grid_search_models.py` script.

This script:
-   Loads and preprocesses the data in the same way as the random search script.
-   Uses `GridSearchCV` to conduct an in-depth search over a predefined grid of hyperparameters for the chosen classifier (the script is pre-configured for RandomForestClassifier but can be adapted for others).
-   Reports the best hyperparameter combination and the model's performance.

To run this script:
```bash
python src/models/grid_search_models.py
```

## Dependencies
This project relies on several popular Python libraries for data science and machine learning, including:
-   `pandas`
-   `numpy`
-   `scikit-learn`
-   `imblearn`
-   `catboost`
-   `lightgbm`
-   `optuna`
-   `matplotlib`
-   `seaborn`

All dependencies are listed in the `requirements.txt` file.
