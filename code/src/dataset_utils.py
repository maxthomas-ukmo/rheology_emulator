import pickle
import math
from torch.utils.data import Dataset, DataLoader, random_split
import torch

from sklearn.preprocessing import StandardScaler

class TorchDataManager:
    def __init__(self, file_path, arguments=None):
        # Initialize the data manager with the file path and arguments
        self.file_path = file_path
        self.batch_size = arguments['batch_size']
        self.val_fraction = arguments['val_fraction']
        self.test_fraction = arguments['test_fraction']
        self.scale = arguments['scale_features']
        self.train_features = arguments['train_features']
        self.train_labels = arguments['train_labels']

        # Load the data from the specified file path
        self.raw_data = self._load()
        self.pairs = FeatureLabelDataset(self.raw_data)

        # Get loaders for training, validation, and testing
        self.train, self.val, self.test = self._get_data_loaders()

        # Print short summary
        self._print_summary()
        
    def _print_summary(self):
        print(f"Data loaded from {self.file_path}")
        print(f"Batch size: {self.batch_size}")
        print(f"Validation fraction: {self.val_fraction}")
        print(f"Test fraction: {self.test_fraction}")
        print(f"Dataset sizes: Train={len(self.train.dataset)}, Val={len(self.val.dataset)}, Test={len(self.test.dataset)}")
    

    def _load(self):
        # with open(self.file_path, 'rb') as f:
        #     return pickle.load(f)
        
        with open(self.file_path, 'rb') as f:
            pairs = pickle.load(f)

        # Get the required features and labels, and convert to a numpy array
        pairs = self._extract_features_labels(pairs)

        return pairs
    
    def _extract_features_labels(self, pairs):
        # Subset to the desired features and labels
        for ipair, pair in enumerate(pairs):
            features = pair[0][self.train_features]
            labels = pair[1][self.train_labels]
            pairs[ipair] = [features.values, labels.values]

        return pairs

    
    def _scale_features(self, pairs):
        scaler = StandardScaler()
        # Fit the scaler on the features
        scaler.fit(pairs[0][0])  # TODO: This assumes first feature is representative, so need to think about this
        # Transform the features
        scaled_features = [scaler.transform(features) for features, _ in pairs]
        # Return pairs with scaled features without converting to torch tensor
        self.scaler = scaler
        pairs = [(scaled_features[i], label) for i, (_, label) in enumerate(pairs)]

    def _make_torch_tensors(self, pairs):
        # Convert data to PyTorch tensors
        tensor_data = [(torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)) for features, label in pairs]
        return tensor_data

    def _split_dataset(self):
        total = len(self.dataset)
        test_len = math.floor(total * self.test_fraction)
        val_len = math.floor(total * self.val_fraction)
        train_len = total - val_len - test_len
        return random_split(self.dataset, [train_len, val_len, test_len])

    def _get_data_loaders(self):

        # Scale data, or not
        if self.scale:
            self._scale_features(self.pairs)

        # Convert pairs to PyTorch tensors
        self.dataset = self._make_torch_tensors(self.pairs)

        # Split the dataset into training, validation, and test sets
        train_set, val_set, test_set = self._split_dataset()

        # Create DataLoader objects for each set
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

class FeatureLabelDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features, label = self.data[idx]
        return features, label
