import pickle 
import torch

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

# import ordinary least squares regression
from sklearn.linear_model import LinearRegression

from dataset_utils import TorchDataManager


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

def main(arguments):
    # Initialize the data manager
    data_manager = TorchDataManager(arguments['pairs_path'], arguments)

    # Define train, val and test datasets
    train_loader = data_manager.train.dataset
    val_loader = data_manager.val.dataset
    # test_loader = data_manager.test.dataset # not currently used

    # TODO: move these to the data manager
    n_features = train_loader[0][0].shape[0]  # Number of features in the dataset
    n_labels = train_loader[0][1].shape[0]  # Number of labels in the dataset

    # Define nn model
    model = define_nn(n_features, n_labels, architecture=None)

    # Define loss function and optimizer
    criterion, optimizer, n_epochs = nn_options(model, architecture=None)

    # TODO: wrap in class
    # Train nn model
    train_losses = []
    val_losses = []

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            #inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                # inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        val_losses.append(val_loss / len(val_loader))

        print(f"Epoch {epoch+1}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

    # Save trained model
    # torch.save(model.state_dict(), arguments['model_save_path'])
    # print(f'Model saved to {arguments["model_save_path"]}')

    plt.figure(figsize=(5, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    model.eval()  # Set model to evaluation mode
    predictions = []
    true_values = []

    with torch.no_grad():  # Disable gradient tracking
        for inputs, targets in val_loader:
            #inputs = inputs.to(device)
            outputs = model(inputs)
            # predictions.append(outputs.cpu())
            # true_values.append(targets.cpu())
            predictions.append(outputs)
            true_values.append(targets)

    # Concatenate all batches into single tensors
    predictions = torch.cat(predictions, dim=0)
    true_values = torch.cat(true_values, dim=0)


    # Fit a linear regression model on the predictions and true values, then plot the trendline
    reg = LinearRegression().fit(true_values, predictions)
    plt.figure(figsize=(8, 8))  
    y_pred = reg.predict(true_values)
    gradient = reg.coef_[0][0]


    plt.scatter(true_values, predictions, s=0.1)
    plt.plot([-5,5], [-5,5], 'r--', label='1:1')  # Diagonal line
    plt.plot(true_values, y_pred, 'r', label='Linear fit: ' + str(gradient))  # Trendline
    plt.xlabel('True Values')
    plt.ylabel('Predictions')

    plt.grid()
    plt.legend()

    plt.show()


if __name__ == "__main__":
    args = {'pairs_path': './pairs.pkl',
            'batch_size': 32,
            'val_fraction': 0.2,
            'test_fraction': 0.1,
            'scale_features': True}
    
    main(args)

