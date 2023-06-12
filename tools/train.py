######### 
# WARNING! Run this file under main directory, not inside of tools! => python tools/train.py
#########

import argparse 
import sys, os
sys.path.append(os.getcwd())
from src import sentiment_dataloader
from src import select_model

def train(trainset_path: str, testset_path: str, model_name: str = "svm", trainset_size: int = 100, testset_size: int = 100, epochs: int = 300, max_feature: int = 1000):
    print(f"trainset_path: {trainset_path}, \ntestset_path: {testset_path}, \nmodel_type: {model_name}")
    
    train_set = sentiment_dataloader(trainset_path, train_size=trainset_size, max_feature=max_feature, is_train=True)
    test_set = sentiment_dataloader(testset_path, train_size=testset_size, vectorizer=train_set.vectorizer, is_train=False)

    model = select_model(model_name, epochs, input_dim=train_set.get_input_size(), hidden_dim=128, output_dim=train_set.get_class_size())
    print(model)
    
    model.train_pipeline(train_set.X, train_set.labels, test_set.X, test_set.labels)
    
    acc_train = model.get_accuracy(train_set.labels, model.predict_pipeline(train_set.X))
    print(f"acc_train: {acc_train}")
    acc_test = model.get_accuracy(test_set.labels, model.predict_pipeline(test_set.X))
    print(f"acc_test: {acc_test}")
    
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentiment Analysis on IMDB Movie Reviews")
    parser.add_argument("--trainset-path", type=str, default="./data/Train.csv", help="Path to the IMDB movie reviews train dataset CSV file")
    parser.add_argument("--valset-path", type=str, default="./data/Valid.csv", help="Path to the IMDB movie reviews test dataset CSV file")
    parser.add_argument("--model-name", "-m", type=str, default="svm", help='Model Name => ["svm", "random_forest", "gru"], default:"svm"')
    parser.add_argument("--train-size", type=int, default=200, help="Trainset size that randomly selecting among all of them randomly")
    parser.add_argument("--test-size", type=int, default=100, help="Testset size that randomly selecting among all of them randomly")
    parser.add_argument("--epochs", type=int, default=300, help="Train epochs")
    parser.add_argument("--max-feature-size", type=int, default=1000, help="Max feature size for Vectorization")
    args = parser.parse_args()
    
    # argparse.Namespace => dict => list of values     
    args = [*vars(args).values()]   
    train(*args)