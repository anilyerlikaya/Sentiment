import argparse 
import sys, os
sys.path.append(os.getcwd())

from sklearn.feature_extraction.text import CountVectorizer
import torch

from src import sentiment_dataloader, select_model, load_model 
from src.data import SentimentDataset
from src.model import BaseSentimentModel

def test(model_path: str = "", model_type: str = "", testset_path: str = "", testset_size: int = 1000, max_feature: int = 1000, \
         vectorizer: CountVectorizer = None, train_test_set: SentimentDataset = None, test_model: BaseSentimentModel = None):
    is_train = test_model != None
   
    model = select_model(model_type, input_dim=max_feature, hidden_dim=128, output_dim=2) if not is_train else test_model
    if not is_train:
        print(model)  
        vectorizer = load_model(model, model_type, model_path)
    dataset = sentiment_dataloader(testset_path, train_size=testset_size, vectorizer=vectorizer, is_train=False) if not is_train else train_test_set
    
    acc_test = model.get_accuracy(dataset.labels, model.predict_pipeline(torch.from_numpy(dataset.X) if model_type == "gru" else dataset.X))
    return acc_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentiment Analysis on IMDB Movie Reviews")
    parser.add_argument("--model-path", "-p", type=str, help="Path of the model that ll be tested")
    parser.add_argument("--model-type", "-m", type=str, help='Model Name => ["svm", "random_forest", "gru"], default:"svm"')
    parser.add_argument("--testset-path", type=str, default="./data/Test.csv", help="Path to the IMDB movie reviews test dataset CSV file")
    parser.add_argument("--test-size", type=int, default=100, help="Testset size that randomly selecting among all of them randomly")
    parser.add_argument("--max-feature-size", type=int, default=1000, help="Max feature size for Vectorization")
    args = parser.parse_args()
    
    # argparse.Namespace => dict => list of values     
    args = [*vars(args).values()]   
    acc_test = test(*args)
    print(f"Test accuracy: {acc_test}")

