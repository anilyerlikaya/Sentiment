import os

from sklearn.feature_extraction.text import CountVectorizer
from .data import SentimentDataset

def sentiment_dataloader(data_path: str, train_size: int = 1000, vectorizer: CountVectorizer = None, max_feature: int = 1000, is_train: bool = False):
    if not os.path.exists(data_path):
        raise RuntimeError("{} not found!".format(data_path))
    
    dataset = SentimentDataset(data_path, train_size=train_size, vectorizer=vectorizer, max_feature=max_feature, is_train=is_train)      
    return dataset