import os 
import pickle
import torch

from .base import BaseSentimentModel
from sklearn.feature_extraction.text import CountVectorizer
from src.utils import create_dir

def save_model(model: BaseSentimentModel, model_type: str, vectorizer: CountVectorizer, save_name: str = "m_model"):
    # check output path if exists
    final_name = os.getcwd() + "/weights"
    create_dir(final_name)
    final_name = final_name + "/" + model_type + "_" + save_name    
    
    if model_type in ["svm", "random_forest"]:
        final_name = final_name + ".sav"
        pickle.dump( {"model": model.model, "vectorizer": vectorizer}, open(final_name, "wb"))
    elif model_type in ["gru"]:
        final_name = final_name + ".pt"
        torch.save({"model": model.state_dict(), "vectorizer": vectorizer}, final_name)
    else:
        raise RuntimeError("Invalid model type for saving model!")

def load_model(model: BaseSentimentModel, model_type: str, save_file: str):
    if not os.path.exists(save_file):
        raise RuntimeError("Model cannot load! Saved model file not found.")
    
    vectorizer: CountVectorizer = None
    if model_type in ["svm", "random_forest"]:
        loaded_package = pickle.load(open(save_file, "rb"))
        model.model = loaded_package["model"]
        vectorizer = loaded_package["vectorizer"] 
    elif model_type in ["gru"]:
        loaded_package = torch.load(save_file)
        vectorizer = loaded_package["vectorizer"] 
        model.load_state_dict(loaded_package["model"])
        model.eval()
    else:
        raise RuntimeError("Invalid model type for loading model file!")
    
    return vectorizer
