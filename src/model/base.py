import numpy as np
from sklearn.metrics import accuracy_score

class BaseSentimentModel(object):
    def __init__(self):
        self.model = None
        pass
    
    def train_pipeline(self, X_train, y_train, X_test, y_test):
        pass
    
    def predict_pipeline(self, X_test):
        pass
    
    @staticmethod
    def get_accuracy(y_true: np.ndarray, y_pred: np.ndarray):
        #return accuracy_score(y_true, y_pred) 
        correct = (y_true == y_pred).sum().item()
        total = y_true.size
        accuracy = correct / total
        return accuracy
    
    