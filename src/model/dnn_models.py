from .base import BaseSentimentModel
import numpy as np
#from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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
        print(f"device: {self.device}, input_size: {self.input_size}, hidden: {self.hidden_units}, output: {self.output_size}")
        
        # layers
        self.embedding = nn.Embedding(self.input_size, self.hidden_units)
        self.gru = nn.GRU(batch_first=True, input_size=self.hidden_units, hidden_size=self.hidden_units//2, bidirectional=True)
        self.activation = nn.ReLU()
        self.fc = nn.Linear(self.hidden_units, self.output_size)
            
        return
    
    def __str__(self):
        return f"GRUDNNMODEL(\n\t{self.embedding}\n\t{self.gru}\n\t{self.fc}\n)"
    
    def forward(self, inputs: torch.Tensor):
        output_embed = self.embedding(inputs)
        output_gru, _ = self.gru(output_embed)
        output_gru = output_gru.permute(1, 0, 2)  
        output_gru = self.activation(output_gru)
        outputs = self.fc(output_gru[-1])              
        return outputs

    def train_pipeline(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):        
        criterion = nn.CrossEntropyLoss(reduction="sum")
        optimizer = optim.Adam(self.parameters(), lr=3e-4)
        
        X_train = torch.from_numpy(X_train)
        y_train = torch.from_numpy(y_train)
        X_test = torch.from_numpy(X_test)

        #for epoch in tqdm(range(self.epochs), desc="GRU-DNN Model Training"):
        for epoch in range(self.epochs):
            self.train()
            epoch_loss = 0.0

            for i in range(0, len(X_train), self.batch_size):
                inputs = X_train[i:i + self.batch_size]
                labels = y_train[i:i + self.batch_size]

                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            train_preds = self.predict_pipeline(X_train)
            test_preds = self.predict_pipeline(X_test)
            train_acc = self.get_accuracy(y_train, train_preds)
            test_acc = self.get_accuracy(y_test, test_preds)
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss / len(X_train)}, train_acc: {train_acc}, test_acc: {test_acc}")
            
        print("Train Completed\n")
        return

    def predict_pipeline(self, X_test: torch.Tensor):
        self.eval()
        with torch.no_grad():
            outputs = self(X_test)
            _, predicted = torch.max(outputs.data, 1)
            return predicted.cpu().numpy()