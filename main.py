from wine_classification import ResultsStorage, ModelTrainer, load_wine_dataset, generate_data_profile, preprocess_data, create_results_folder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier  
from sklearn.neighbors import KNeighborsClassifier 
import logging

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

    # Train and tune models
    lr = trainer.train_model(LogisticRegression(max_iter=1000), "Logistic Regression", X_train, y_train)
    dt = trainer.train_model(DecisionTreeClassifier(), "Decision Tree", X_train, y_train)
    rf = trainer.train_model(RandomForestClassifier(), "Random Forest", X_train, y_train)
    knn = trainer.train_model(KNeighborsClassifier(), "K-Nearest Neighbors", X_train, y_train)

    trainer.tune_model(LogisticRegression(max_iter=1000), "Logistic Regression", X_train, y_train, {'C': [0.001, 0.01, 0.1, 1, 10, 100]})
    trainer.tune_model(DecisionTreeClassifier(), "Decision Tree", X_train, y_train, {'max_depth': [3, 5, 7, None], 'min_samples_split': [2, 5, 10]})
    trainer.tune_model(RandomForestClassifier(), "Random Forest", X_train, y_train, {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7, None]})
    trainer.tune_model(KNeighborsClassifier(), "K-Nearest Neighbors", X_train, y_train, {'n_neighbors': [3, 5, 7, 10]})

    results_storage.close_connection()
    logging.info("Wine classification project completed.")