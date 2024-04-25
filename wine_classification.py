import logging
import sqlite3
import os
import pickle
import joblib
from datetime import datetime
import json
from sklearn.datasets import load_wine
import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

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

class ModelTrainer:
    def __init__(self, results_storage, results_folder):
        self.results_storage = results_storage
        self.results_folder = results_folder

    def train_model(self, model, model_name, X_train, y_train):
        logging.info(f"Training {model_name} model...")
        scores = cross_val_score(model, X_train, y_train, cv=5)
        mean_accuracy = scores.mean()
        logging.info(f"{model_name} Mean Accuracy: {mean_accuracy}")
        self.results_storage.insert_result(model_name, "Cross-validation", mean_accuracy)
        self.save_model(model, model_name)
        return mean_accuracy

    def tune_model(self, model, model_name, X_train, y_train, param_grid):
        logging.info(f"Tuning {model_name} model...")
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        self.results_storage.insert_result(model_name, "Hyperparameter Tuning", best_params)
        print(f"Best Parameters for {model_name}:", best_params)

    def save_model(self, model, model_name):
        os.makedirs(self.results_folder, exist_ok=True)
        with open(os.path.join(self.results_folder, f'{model_name}.pkl'), 'wb') as f:
            pickle.dump(model, f)
        joblib.dump(model, os.path.join(self.results_folder, f'{model_name}.joblib'))
        print(f"Model '{model_name}' saved successfully.")

def load_wine_dataset():
    wine_data = load_wine()
    df = pd.DataFrame(data=wine_data.data, columns=wine_data.feature_names)
    df['target'] = wine_data.target
    return df

def handle_missing_values(df):
    if df.isnull().sum().sum() > 0:
        df.fillna(method='ffill', inplace=True)
    logging.info("Missing values handled.")

def generate_data_profile(df):
    profile = ProfileReport(df, title="Wine Dataset Profiling Report", explorative=True)
    profile.to_file("wine_dataset_profile.html")
    logging.info("Data profiling report generated.")

def preprocess_data(df):
    handle_missing_values(df)
    scaler = StandardScaler()
    df[df.columns[:-1]] = scaler.fit_transform(df[df.columns[:-1]])
    X = df.drop('target', axis=1)
    y = df['target']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def create_results_folder():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    folder_path = os.path.join(base_dir, 'results', current_datetime)
    os.makedirs(folder_path, exist_ok=True)
    logging.info(f"Results folder created: {folder_path}")
    return folder_path

if __name__ == "__main__":
    logging.basicConfig(filename='wine_classification.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting wine classification project...")

    results_storage = ResultsStorage()
    results_storage.create_table()
    results_folder = create_results_folder()

    wine_df = load_wine_dataset()
    generate_data_profile(wine_df)
    X_train, X_test, y_train, y_test = preprocess_data(wine_df)

    trainer = ModelTrainer(results_storage, results_folder)

    lr = LogisticRegression(max_iter=1000)
    trainer.train_model(lr, "Logistic Regression", X_train, y_train)

    dt = DecisionTreeClassifier()
    trainer.train_model(dt, "Decision Tree", X_train, y_train)

    rf = RandomForestClassifier()
    trainer.train_model(rf, "Random Forest", X_train, y_train)

    knn = KNeighborsClassifier()
    trainer.train_model(knn, "K-Nearest Neighbors", X_train, y_train)

    trainer.tune_model(lr, "Logistic Regression", X_train, y_train, {'C': [0.001, 0.01, 0.1, 1, 10, 100]})
    trainer.tune_model(dt, "Decision Tree", X_train, y_train, {'max_depth': [3, 5, 7, None], 'min_samples_split': [2, 5, 10]})
    trainer.tune_model(rf, "Random Forest", X_train, y_train, {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7, None]})
    trainer.tune_model(knn, "K-Nearest Neighbors", X_train, y_train, {'n_neighbors': [3, 5, 7, 10]})

    results_storage.close_connection()
    logging.info("Wine classification project completed.")


