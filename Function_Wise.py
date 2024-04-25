import logging
import sqlite3
import os
import pickle
import joblib
from sklearn.datasets import load_wine
import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import json
from datetime import datetime

logging.basicConfig(filename='wine_classification.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ResultsStorage:
    def __init__(self, db_name='wine_classification_results.db'):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()

    def create_table(self):
        schema = '''CREATE TABLE IF NOT EXISTS results
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     model TEXT,
                     type TEXT,
                     result TEXT)'''
        self.cursor.execute(schema)
        self.conn.commit()

    def insert_result(self, model, result_type, result):
        if isinstance(result, dict):
            result = json.dumps(result)
        self.cursor.execute('''INSERT INTO results (model, type, result)
                               VALUES (?, ?, ?)''', (model, result_type, result))
        self.conn.commit()

    def close_connection(self):
        self.conn.close()

def load_wine_dataset():
    wine_data = load_wine()
    df = pd.DataFrame(data=wine_data.data, columns=wine_data.feature_names)
    df['target'] = wine_data.target
    return df


def handle_missing_values(df):
    # Check for missing values
    if df.isnull().sum().sum() == 0:
        print("No missing values found.")
    else:
        # Handle missing values
        df.fillna(method='ffill', inplace=True)  # Example: Forward fill missing values
        print("Missing values handled.")


def generate_data_profile(df):
    profile = ProfileReport(df, title="Wine Dataset Profiling Report", explorative=True)
    profile.to_file("wine_dataset_profile.html")
    print("Data profiling report generated.")


def preprocess_data(df):
    # Handling missing values
    if df.isnull().sum().sum() > 0:
        df.fillna(method='ffill', inplace=True)  # Example: Forward fill missing values

    # Feature scaling or normalization
    scaler = StandardScaler()
    df[df.columns[:-1]] = scaler.fit_transform(df[df.columns[:-1]])

    # Splitting dataset into training and testing sets
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train, y_train, results_storage):
    logging.info("Training Logistic Regression model...")
    lr = LogisticRegression(max_iter=1000)
    scores = cross_val_score(lr, X_train, y_train, cv=5)
    mean_accuracy = scores.mean()
    logging.info("Logistic Regression Mean Accuracy: %f", mean_accuracy)
    results_storage.insert_result("Logistic Regression", "Cross-validation", mean_accuracy)
    
    # Save trained model
    save_model_with_pickle(lr, "logistic_regression", results_folder)
    save_model_with_joblib(lr, "logistic_regression", results_folder)

    return mean_accuracy


def train_decision_tree(X_train, y_train, results_storage):
    logging.info("Training Decision Tree model...")
    dt = DecisionTreeClassifier()
    scores = cross_val_score(dt, X_train, y_train, cv=5)
    mean_accuracy = scores.mean()
    logging.info("Decision Tree Mean Accuracy: %f", mean_accuracy)
    results_storage.insert_result("Decision Tree", "Cross-validation", mean_accuracy)
    
    # Save trained model
    save_model_with_pickle(dt, "decision_tree", results_folder)
    save_model_with_joblib(dt, "decision_tree", results_folder)

    return mean_accuracy


def train_random_forest(X_train, y_train, results_storage):
    logging.info("Training Random Forest model...")
    rf = RandomForestClassifier()
    scores = cross_val_score(rf, X_train, y_train, cv=5)
    mean_accuracy = scores.mean()
    logging.info("Random Forest Mean Accuracy: %f", mean_accuracy)
    results_storage.insert_result("Random Forest", "Cross-validation", mean_accuracy)
    
    # Save trained model
    save_model_with_pickle(rf, "random_forest", results_folder)
    save_model_with_joblib(rf, "random_forest", results_folder)

    return mean_accuracy


def train_knn(X_train, y_train, results_storage):
    logging.info("Training K-Nearest Neighbors model...")
    knn = KNeighborsClassifier()
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    mean_accuracy = scores.mean()
    logging.info("K-Nearest Neighbors Mean Accuracy: %f", mean_accuracy)
    results_storage.insert_result("K-Nearest Neighbors", "Cross-validation", mean_accuracy)
    
    # Save trained model
    save_model_with_pickle(knn, "knn", results_folder)
    save_model_with_joblib(knn, "knn", results_folder)

    return mean_accuracy


def tune_logistic_regression(X_train, y_train, results_storage):
    lr = LogisticRegression(max_iter=1000)
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    results_storage.insert_result("Logistic Regression", "Hyperparameter Tuning", best_params)
    print("Best Parameters for Logistic Regression:", best_params)


def tune_decision_tree(X_train, y_train, results_storage):
    dt = DecisionTreeClassifier()
    param_grid = {'max_depth': [3, 5, 7, None], 'min_samples_split': [2, 5, 10]}
    grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    results_storage.insert_result("Decision Tree", "Hyperparameter Tuning", best_params)
    print("Best Parameters for Decision Tree:", best_params)


def tune_random_forest(X_train, y_train, results_storage):
    rf = RandomForestClassifier()
    param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7, None]}
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    results_storage.insert_result("Random Forest", "Hyperparameter Tuning", best_params)
    print("Best Parameters for Random Forest:", best_params)


def tune_knn(X_train, y_train, results_storage):
    knn = KNeighborsClassifier()
    param_grid = {'n_neighbors': [3, 5, 7, 10]}
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    results_storage.insert_result("K-Nearest Neighbors", "Hyperparameter Tuning", best_params)
    print("Best Parameters for K-Nearest Neighbors:", best_params)


def save_model_with_pickle(model, model_name, folder_path):
    # Ensure folder exists
    os.makedirs(folder_path, exist_ok=True)
    # Save model using pickle
    with open(os.path.join(folder_path, f'{model_name}.pkl'), 'wb') as f:
        pickle.dump(model, f)
    print(f"Model '{model_name}' saved with pickle successfully.")

def save_model_with_joblib(model, model_name, folder_path):
    # Ensure folder exists
    os.makedirs(folder_path, exist_ok=True)
    # Save model using joblib
    joblib.dump(model, os.path.join(folder_path, f'{model_name}.joblib'))
    print(f"Model '{model_name}' saved with joblib successfully.")

def create_results_folder():
    # Define the base directory where the results folder will be created
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Create folder with current date and time
    current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    folder_path = os.path.join(base_dir, 'results', current_datetime)
    os.makedirs(folder_path, exist_ok=True)
    print(f"Results folder created: {folder_path}")
    return folder_path

# Usage
logging.info("Starting wine classification project...")
results_storage = ResultsStorage()
results_storage.create_table()
results_folder = create_results_folder()
wine_df = load_wine_dataset()
handle_missing_values(wine_df)
generate_data_profile(wine_df)
X_train, X_test, y_train, y_test = preprocess_data(wine_df)
train_logistic_regression(X_train, y_train, results_storage)
train_decision_tree(X_train, y_train, results_storage)
train_random_forest(X_train, y_train, results_storage)
train_knn(X_train, y_train, results_storage)
tune_logistic_regression(X_train, y_train, results_storage)
tune_decision_tree(X_train, y_train, results_storage)
tune_random_forest(X_train, y_train, results_storage)
tune_knn(X_train, y_train, results_storage)
results_storage.close_connection()
logging.info("Wine classification project completed.")

