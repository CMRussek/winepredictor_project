from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

class Logger:
    def __init__(self, filename='wine_classification.log', level=logging.INFO):
        self.logger = logging.getLogger('WineClassification')
        self.logger.setLevel(level)
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.file_handler = logging.FileHandler(filename)
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.file_handler)

    def log_info(self, message):
        self.logger.info(message)

    def log_warning(self, message):
        self.logger.warning(message)

    def log_error(self, message):
        self.logger.error(message)

    def log_critical(self, message):
        self.logger.critical(message)

class ModelTrainer:
    def __init__(self, logger):
        self.logger = logger

    def train_logistic_regression(self, X_train, y_train):
        self.logger.log_info("Training Logistic Regression model...")
        lr = LogisticRegression(max_iter=1000)
        scores = cross_val_score(lr, X_train, y_train, cv=5)
        mean_accuracy = scores.mean()
        self.logger.log_info("Logistic Regression Mean Accuracy: %f", mean_accuracy)
        return mean_accuracy

    def train_decision_tree(self, X_train, y_train):
        self.logger.log_info("Training Decision Tree model...")
        dt = DecisionTreeClassifier()
        scores = cross_val_score(dt, X_train, y_train, cv=5)
        mean_accuracy = scores.mean()
        self.logger.log_info("Decision Tree Mean Accuracy: %f", mean_accuracy)
        return mean_accuracy

    def train_random_forest(self, X_train, y_train):
        self.logger.log_info("Training Random Forest model...")
        rf = RandomForestClassifier()
        scores = cross_val_score(rf, X_train, y_train, cv=5)
        mean_accuracy = scores.mean()
        self.logger.log_info("Random Forest Mean Accuracy: %f", mean_accuracy)
        return mean_accuracy

    def train_knn(self, X_train, y_train):
        self.logger.log_info("Training K-Nearest Neighbors model...")
        knn = KNeighborsClassifier()
        scores = cross_val_score(knn, X_train, y_train, cv=5)
        mean_accuracy = scores.mean()
        self.logger.log_info("K-Nearest Neighbors Mean Accuracy: %f", mean_accuracy)
        return mean_accuracy

def main():
    # Initialize Logger
    logger = Logger()

    # Starting wine classification project
    logger.log_info("Starting wine classification project...")

    # Load data
    wine_data = load_wine()
    X = wine_data.data
    y = wine_data.target

    # Preprocess data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize ModelTrainer
    model_trainer = ModelTrainer(logger)

    # Train models
    accuracy_lr = model_trainer.train_logistic_regression(X_train, y_train)
    accuracy_dt = model_trainer.train_decision_tree(X_train, y_train)
    accuracy_rf = model_trainer.train_random_forest(X_train, y_train)
    accuracy_knn = model_trainer.train_knn(X_train, y_train)

    # Further steps...

    # Completing wine classification project
    logger.log_info("Wine classification project completed.")

if __name__ == "__main__":
    main()
