from .base import BaseSentimentModel
import numpy as np

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

class SVMModel(BaseSentimentModel):
    def __init__(self, epochs: int = 300, verbose: bool = False):
        super().__init__()
        self.epochs = epochs
        self.model = SVC(kernel="rbf", tol=1e-4, max_iter=self.epochs, verbose=False)

    def __str__(self):
        return f"{self.model}"

    def train_pipeline(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
        self.model.fit(X_train, y_train)
        print("Train Completed\n")

    def predict_pipeline(self, X_test: np.ndarray):
        return self.model.predict(X_test)


class RandomForestModel(BaseSentimentModel):
    def __init__(self, verbose: bool = False):
        super().__init__()
        self.model = RandomForestClassifier(verbose=verbose)
        
    def __str__(self):
        return f"{self.model}"

    def train_pipeline(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
        self.model.fit(X_train, y_train)
        print("Train Completed\n")

    def predict_pipeline(self, X_test: np.ndarray):
        return self.model.predict(X_test)