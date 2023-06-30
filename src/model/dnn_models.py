from .base import BaseSentimentModel
import copy
import numpy as np
#from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Train Plot for Acc&Loss
from src.utils import DrawTrain

# One can also set device for model and criterion
# For my case, i work on cpu
class GRUDNNModel(nn.Module, BaseSentimentModel):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, epochs: int = 300, batch_size: int = 8, device: str = "cpu"):
        super(GRUDNNModel, self).__init__()
        
        # params
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.input_size = input_dim
        self.hidden_units = hidden_dim
        self.output_size = output_dim
        #print(f"device: {self.device}, input_size: {self.input_size}, hidden: {self.hidden_units}, output: {self.output_size}")
        
        # layers
        self.embedding = nn.Embedding(self.input_size, self.hidden_units)
        self.gru = nn.GRU(batch_first=True, input_size=self.hidden_units, hidden_size=self.hidden_units//2, bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.hidden_units, self.output_size)
        self.sigmoid = nn.Sigmoid()
            
        # plotter
        self.plotter = DrawTrain()
            
        self.setup()
        return
    
    def __str__(self):
        return f"GRUDNNMODEL(\n\t{self.embedding}\n\t{self.gru}\n\t{self.fc}\n)"
    
    def setup(self):
        self.criterion = nn.BCELoss(reduction="sum")
        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)
        return
    
    def forward(self, inputs: torch.Tensor):
        batch_size = inputs.shape[0]   
        output_embed = self.embedding(inputs)
        
        output_gru, _ = self.gru(output_embed)
        output_gru = output_gru.contiguous().view(-1, self.hidden_units)
        outputs = self.dropout(output_gru)
        
        outputs = self.fc(outputs)
        outputs = self.sigmoid(outputs)
        outputs = outputs.view(batch_size, -1)[:, -1] # last batch
        return outputs.squeeze()

    def train_pipeline(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):        
        X_train = torch.from_numpy(X_train)
        y_train_torch = torch.from_numpy(y_train).float()
        X_test = torch.from_numpy(X_test)
        y_test_torch = torch.from_numpy(y_test).float()

        # keep best
        best_state_dict = copy.deepcopy(self.state_dict())
        best_acc_epoch = -1
        best_test_acc = -1.0

        #for epoch in tqdm(range(self.epochs), desc="GRU-DNN Model Training"):
        for epoch in range(self.epochs):
            self.train()
            epoch_loss = 0.0

            for i in range(0, len(X_train), self.batch_size):
                inputs = X_train[i:i + self.batch_size]
                labels = y_train_torch[i:i + self.batch_size]

                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            train_preds = self.predict_pipeline(X_train)
            test_preds, test_loss = self.predict_pipeline(X_test, y_test_torch)
            train_loss = epoch_loss / len(X_train)
            train_acc = self.get_accuracy(y_train, train_preds)
            test_acc = self.get_accuracy(y_test, test_preds)            
            self.plotter.update(train_loss, test_loss, train_acc, test_acc) # plot
            print(f"Epoch {epoch + 1}/{self.epochs}, Train Loss: {train_loss}, Test Loss: {test_loss}, train_acc: {train_acc}, test_acc: {test_acc}")      
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_acc_epoch = epoch
                best_state_dict = copy.deepcopy(self.state_dict())
            
        print("Train Completed\n")
        print(f"Acquired best accuracy in epoch {best_acc_epoch} and test_acc is {best_test_acc} => Loaded current model state_dict from the best one.")
        self.load_state_dict(best_state_dict)        
        self.plotter.show()     # wait
        
        return

    def predict_pipeline(self, X_test: torch.Tensor, y_test: torch.Tensor = None):
        self.eval()
        with torch.no_grad():
            outputs = self(X_test)
            preds = torch.round(outputs)
            if y_test is not None:
                test_loss = self.criterion(outputs, y_test) / len(y_test)
                return preds.cpu().numpy(), test_loss
            return preds.cpu().numpy()