# config.py

class Config:
    # File paths
    DATA_FILE_PATH = 'data/wine_data.csv'
    LOG_FILE_PATH = 'logs/wine_classification.log'
    MODEL_SAVE_PATH = 'models/'

    # Model parameters
    LOGISTIC_REGRESSION_PARAMS = {'max_iter': 1000, 'C': 1.0}
    DECISION_TREE_PARAMS = {'max_depth': None, 'min_samples_split': 2}
    RANDOM_FOREST_PARAMS = {'n_estimators': 100, 'max_depth': None}
    KNN_PARAMS = {'n_neighbors': 5}


# main.py

#Example usage

#from config import Config

#print(Config.DATA_FILE_PATH)
#print(Config.LOGISTIC_REGRESSION_PARAMS)