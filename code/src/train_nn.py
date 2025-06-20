import pickle 
import torch
import yaml


import matplotlib.pyplot as plt
from pathlib import Path
import torch.nn as nn
import torch.optim as optim


# import ordinary least squares regression
from sklearn.linear_model import LinearRegression

from .dataset_utils import TorchDataManager

def nn_layer_list(config_path='../configs/nn_architecture/base.yaml'):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)['model']
    layer_list = []
    for item in config:
        layer_list.append([config[item]['type'], config[item]['args']])
    return layer_list

def match_io_dims(layer_list, n_features, n_labels):
    layer_list[0][1][0] = n_features  # Set input dimension of the first layer
    layer_list[-1][1][1] = n_labels  # Set output dimension of the last layer
    return layer_list

def build_model_from_layers(layer_list):
    # Create a list to hold the layers
    layers = []

    # Iterate through the layer definitions
    for layer_type, args in layer_list:
        # Get the layer class from nn
        layer_class = getattr(torch.nn, layer_type)
        # Instantiate the layer with the provided arguments
        layers.append(layer_class(*args))

    # Create the neural network using nn.Sequential
    model = torch.nn.Sequential(*layers)

    # Print the model
    print('Neural net architecture:')
    print(model)

    return model

def nn_options(model, parameters='../configs/parameters/nn_base.yaml'):
    """
    Define the loss function, optimizer, and number of epochs based on a YAML configuration.
    """
    if parameters is None:
        raise ValueError("An architecture YAML file must be provided.")

    # Load the YAML configuration
    with open(parameters, "r") as f:
        config = yaml.safe_load(f)

    # Define the loss function
    loss_type = config.get("loss", "MSELoss")  # Default to MSELoss if not specified
    criterion = getattr(nn, loss_type)()

    # Define the optimizer
    optimizer_type = config.get("optimizer", "Adam")  # Default to Adam if not specified
    lr = config.get("learning_rate", 0.001)  # Default learning rate
    optimizer_class = getattr(optim, optimizer_type)
    optimizer = optimizer_class(model.parameters(), lr=lr)

    # Define the number of epochs
    n_epochs = config.get("epochs", 10)  # Default to 10 epochs

    return criterion, optimizer, n_epochs



class NNCapsule:
    def __init__(self, arguments):
        self.arguments = arguments

        # Load data
        self.data_manager = TorchDataManager(arguments['pairs_path'], arguments)
        self.train_loader = self.data_manager.train.dataset
        self.val_loader = self.data_manager.val.dataset
        self.n_features = self.train_loader[0][0].shape[1]
        self.n_labels = self.train_loader[0][1].shape[1]
        self.n_observations = self.train_loader[0][0].shape[0]
        self.scaler = self.data_manager.scaler

        # Define model
        self.architecture = arguments['architecture']
        #self.parameters = arguments['parameters']
        # TODO: this is clunky and params should be wrapped up into arguments
        self.parameters = '../configs/training/' + arguments['training_cfg'] + '.yaml'
        self.model = self._define_nn()
        # TODO: split the below up so that they're called separately, or do some order agnostic unpacking of all the parameters
        self.criterion, self.optimizer, self.n_epochs = nn_options(self.model, self.parameters)
        self.train_losses = []
        self.val_losses = []

    def _define_nn(self):
        layer_list = nn_layer_list(self.architecture)
        layer_list = match_io_dims(layer_list, self.n_features, self.n_labels)
        model = build_model_from_layers(layer_list)
        return model


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
        plt.savefig(self.arguments['results_path'] + 'train_losses.png')
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

        # Calculate MSE
        mse = torch.nn.functional.mse_loss(predictions, true_values)

        plt.title(f'MSE: {mse:.2e}, Gradient: {gradient:.4f}')
        plt.scatter(true_values, predictions, s=0.1)
        plt.plot([-5,5], [-5,5], 'r--', label='1:1')
        plt.plot(true_values, y_pred, 'r', label='Linear fit: ' + str(gradient))
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.grid()
        plt.legend()
        plt.savefig(self.arguments['results_path'] + 'true_pred.png')
            

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

