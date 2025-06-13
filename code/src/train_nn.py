import pickle 
import torch

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

# import ordinary least squares regression
from sklearn.linear_model import LinearRegression

from .dataset_utils import TorchDataManager


def define_nn(n_features, n_labels, architecture=None):
    # TODO: Replace this placeholder ---------
    class SimpleNN(nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(2, 64)
            self.fc2 = nn.Linear(64, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    model = SimpleNN()
    # ---------------------------
    return model

def nn_options(model, architecture=None):
    # TODO: Replace this placeholder ---------
    criterion = nn.MSELoss()  # Mean Squared Error Loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer
    n_epochs = 10  # Number of epochs for training
    return criterion, optimizer, n_epochs
    # ---------------------------

class NNCapsule:
    def __init__(self, arguments):
        self.arguments = arguments

        # Load data
        self.data_manager = TorchDataManager(arguments['pairs_path'], arguments)
        self.train_loader = self.data_manager.train.dataset
        self.val_loader = self.data_manager.val.dataset
        self.n_features = self.train_loader[0][0].shape[0]
        self.n_labels = self.train_loader[0][1].shape[0]
        self.scaler = self.data_manager.scaler

        # Define model
        self.model = define_nn(self.n_features, self.n_labels, architecture=None)
        self.criterion, self.optimizer, self.n_epochs = nn_options(self.model, architecture=None)
        self.train_losses = []
        self.val_losses = []

    def train(self):
        for epoch in range(self.n_epochs):
            self.model.train()
            running_loss = 0.0
            for inputs, targets in self.train_loader:
                # inputs, targets = inputs.to(device), targets.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            self.train_losses.append(running_loss / len(self.train_loader))

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in self.val_loader:
                    # inputs, targets = inputs.to(device), targets.to(device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    val_loss += loss.item()
            self.val_losses.append(val_loss / len(self.val_loader))

            print(f"Epoch {epoch+1}, Train Loss: {self.train_losses[-1]:.4f}, Val Loss: {self.val_losses[-1]:.4f}")

    def plot_train_losses(self, train_losses, val_losses):
        fig = plt.figure(figsize=(5, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        # TODO: replace show with some saving option
        plt.show()
        return fig
    
    def ytrue_ypred(self, loader):
        predictions = []
        true_values = []
        with torch.no_grad():  # Disable gradient tracking
            for inputs, targets in loader:
                #inputs = inputs.to(device)
                outputs = self.model(inputs)
                # predictions.append(outputs.cpu())
                # true_values.append(targets.cpu())
                predictions.append(outputs)
                true_values.append(targets)

        # Concatenate all batches into single tensors
        predictions = torch.cat(predictions, dim=0)
        true_values = torch.cat(true_values, dim=0)

        return true_values, predictions
    
    def plot_ytrue_ypred(self, loader):
        true_values, predictions = self.ytrue_ypred(loader)

        # Fit a linear regression model on the predictions and true values, then plot the trendline
        reg = LinearRegression().fit(true_values, predictions)
        plt.figure(figsize=(8, 8))  
        y_pred = reg.predict(true_values)
        gradient = reg.coef_[0][0]

        plt.scatter(true_values, predictions, s=0.1)
        plt.plot([-5,5], [-5,5], 'r--', label='1:1')
        plt.plot(true_values, y_pred, 'r', label='Linear fit: ' + str(gradient))
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.grid()
        plt.legend()
        # TODO: replace show with some saving option
        plt.show()
            

def train_save_eval(arguments):

    nn_capsule = NNCapsule(arguments)

    nn_capsule.train()

    nn_capsule.plot_train_losses(nn_capsule.train_losses, nn_capsule.val_losses)
    nn_capsule.plot_ytrue_ypred(nn_capsule.val_loader)




if __name__ == "__main__":
    args = {'pairs_path': './pairs.pkl',
            'batch_size': 32,
            'val_fraction': 0.2,
            'test_fraction': 0.1,
            'scale_features': True}
    
    train_save_eval(args)

