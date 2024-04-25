from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from Data_Handler_class import DataHandler


class ModelTrainer:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        

    def train_logistic_regression(self):
        lr = LogisticRegression(max_iter=1000)
        scores = cross_val_score(lr, self.X_train, self.y_train, cv=5)
        print("Logistic Regression Mean Accuracy:", scores.mean())

    def train_decision_tree(self):
        dt = DecisionTreeClassifier()
        scores = cross_val_score(dt, self.X_train, self.y_train, cv=5)
        print("Decision Tree Mean Accuracy:", scores.mean())

    def train_random_forest(self):
        rf = RandomForestClassifier()
        scores = cross_val_score(rf, self.X_train, self.y_train, cv=5)
        print("Random Forest Mean Accuracy:", scores.mean())

    def train_knn(self):
        knn = KNeighborsClassifier()
        scores = cross_val_score(knn, self.X_train, self.y_train, cv=5)
        print("K-Nearest Neighbors Mean Accuracy:", scores.mean())

    def tune_logistic_regression(self):
        lr = LogisticRegression(max_iter=1000)
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
        grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(self.X_train, self.y_train)
        print("Best Parameters for Logistic Regression:", grid_search.best_params_)

    def tune_decision_tree(self):
        dt = DecisionTreeClassifier()
        param_grid = {'max_depth': [3, 5, 7, None], 'min_samples_split': [2, 5, 10]}
        grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(self.X_train, self.y_train)
        print("Best Parameters for Decision Tree:", grid_search.best_params_)

    def tune_random_forest(self):
        rf = RandomForestClassifier()
        param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7, None]}
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(self.X_train, self.y_train)
        print("Best Parameters for Random Forest:", grid_search.best_params_)

    def tune_knn(self):
        knn = KNeighborsClassifier()
        param_grid = {'n_neighbors': [3, 5, 7, 10]}
        grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(self.X_train, self.y_train)
        print("Best Parameters for K-Nearest Neighbors:", grid_search.best_params_)

# Usage
data_handler = DataHandler()
X_train, y_train = data_handler.get_train_data()
if X_train is not None and y_train is not None:
    model_trainer = ModelTrainer(X_train, y_train)
else:
    print("Error: Data not available for training.")

model_trainer = ModelTrainer(X_train, y_train)
model_trainer.train_logistic_regression()
model_trainer.train_decision_tree()
model_trainer.train_random_forest()
model_trainer.train_knn()
model_trainer.tune_logistic_regression()
model_trainer.tune_decision_tree()
model_trainer.tune_random_forest()
model_trainer.tune_knn()