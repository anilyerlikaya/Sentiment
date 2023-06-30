import numpy as np
import matplotlib.pyplot as plt

class DrawTrain:    
    def __init__(self, ):
        self.fig, self.ax = plt.subplots(1, 2, figsize=(10, 4))
        plt.ion() # Enable interactive mode

        # data holders
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []

        return
    
    def update(self, train_loss: float, test_loss: float, train_acc: float, test_acc: float):
        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss)
        self.train_accuracies.append(train_acc)
        self.test_accuracies.append(test_acc)
        
        self.ax[0].cla()
        self.ax[0].plot(self.train_losses, label='Train Loss')
        self.ax[0].plot(self.test_losses, label='Test Loss')
        self.ax[0].set_xlabel('Iteration')
        self.ax[0].set_ylabel('Loss')
        self.ax[0].legend()
        
        self.ax[1].cla()
        self.ax[1].plot(self.train_accuracies, label='Train Accuracy')
        self.ax[1].plot(self.test_accuracies, label='Test Accuracy')
        self.ax[1].set_xlabel('Iteration')
        self.ax[1].set_ylabel('Accuracy')
        self.ax[1].legend()
        
        plt.tight_layout()
        plt.pause(0.001)    
        return
    
    def show(self):
        plt.ioff()
        plt.show()