from .ml_models import SVMModel, RandomForestModel
from .dnn_models import GRUDNNModel

# Factory code for selecting a model
def select_model(model_name, epochs: int, input_dim: int = 100, hidden_dim: int = 256, output_dim: int = 2, batch_size: int = 8, verbose: bool = False):
    if model_name == "svm":
        return SVMModel(epochs, verbose=verbose)
    elif model_name == "random_forest":
        return RandomForestModel(verbose=verbose)
    elif model_name == "gru":
        return GRUDNNModel(input_dim, hidden_dim, output_dim, epochs=epochs, batch_size=8)
    else:
        raise ValueError("Invalid model name")