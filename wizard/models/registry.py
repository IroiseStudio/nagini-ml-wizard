from typing import Dict, Callable
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# Builders return an estimator with reasonable defaults for HF CPU
CLASSIFIERS: Dict[str, Callable[[], object]] = {
    "Decision Tree": lambda: DecisionTreeClassifier(random_state=42),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
    "MLP":            lambda: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42),
    "SVM":            lambda: SVC(probability=True, kernel="rbf", C=1.0, gamma="scale", random_state=42),
    "Logistic Regression": lambda: LogisticRegression(max_iter=500, n_jobs=None),
    "Naive Bayes":    lambda: GaussianNB(),
    "KNN":            lambda: KNeighborsClassifier(n_neighbors=7),
}

REGRESSORS: Dict[str, Callable[[], object]] = {
    "Decision Tree": lambda: DecisionTreeRegressor(random_state=42),
    "Random Forest": lambda: RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42),
    "MLP":            lambda: MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=400, random_state=42),
    "SVM":            lambda: SVR(kernel="rbf", C=1.0, gamma="scale"),
    "Linear Regression": lambda: LinearRegression(),
    "Ridge":          lambda: Ridge(alpha=1.0, random_state=42),
    "KNN":            lambda: KNeighborsRegressor(n_neighbors=7),
}
