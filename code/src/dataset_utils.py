import pickle
import math
from torch.utils.data import Dataset, DataLoader, random_split

class TorchDataManager:
    def __init__(self, file_path, arguments=None):
        # Initialize the data manager with the file path and arguments
        self.file_path = file_path
        self.batch_size = arguments['batch_size']
        self.val_fraction = arguments['val_fraction']
        self.test_fraction = arguments['test_fraction']

        # Load the data from the specified file path
        self.data = self._load_data()
        self.dataset = FeatureLabelDataset(self.data)

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
    

    def _load_data(self):
        with open(self.file_path, 'rb') as f:
            return pickle.load(f)

    def _split_dataset(self):
        total = len(self.dataset)
        test_len = math.floor(total * self.test_fraction)
        val_len = math.floor(total * self.val_fraction)
        train_len = total - val_len - test_len
        return random_split(self.dataset, [train_len, val_len, test_len])

    def _get_data_loaders(self):
        train_set, val_set, test_set = self._split_dataset()

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
