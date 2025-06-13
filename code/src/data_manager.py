import random
import os
import pickle

import xarray as xr
import pandas as pd

from pathlib import Path


class DataManager:
    def __init__(self, raw_path, interim_path, pairs_path, arguments=None):
        # Define paths for data. These are either loaded in or created
        self.raw_path = Path(raw_path)
        self.interim_path = Path(interim_path)
        self.pairs_path = Path(pairs_path)

        # Arguments for processing the data. These should be supplied as the defaults are just for testing.
        self.arguments = arguments if arguments is not None else self.default_arguments()

        # Load or prepare the data
        self.load_or_prepare()

        # TODO: decide where and how to set seed
        # Set the random seed for reproducibility
        # random.seed(self.arguments['seed'])

    def default_arguments(self):
        return {'subset_region': 'Arctic',
                'inputs': ['siconc', 'sivelv'],
                'features': ['siconc', 'sivelv'],
                'labels': ['sivelv'],
                'seed': 0}

    def load_or_prepare(self):
        # Step 1: Check raw data
        if not self.raw_path.exists():
            raise FileNotFoundError(f"Raw data not found at {self.raw_path}")
        else:
            print(f"Loading raw data from {self.raw_path}")
            self.raw_data = self._load_netcdf(self.raw_path)

        # Step 2: Check or create interim data
        if self.interim_path.exists():
            print(f'Loading interim data from {self.interim_path}')
            interim_data = self._load_netcdf(self.interim_path)
        else:
            print(f'Creating interim data from {self.raw_path}')
            print(f'Using arguments: {self.arguments}')
            interim_data = self._create_interim()
            self._save_netcdf(interim_data, self.interim_path)
        self.interim_data = interim_data
        print(f'Interim data loaded with shape: {interim_data.sizes}')
        print(f'Interim data variables: {interim_data.data_vars}')

        # Step 3: Check or create model input
        if self.pairs_path.exists():
            print(f'Loading model input data from {self.pairs_path}')
            pairs = self._load_pickle(self.pairs_path)
        else:
            print(f'Creating mode input data from {self.interim_path}')
            print(f'Using arguments: {self.arguments}')
            pairs = self._create_model_input()
            self._save_pickle(pairs, self.pairs_path)
        self.pairs = pairs
        print(f'Model input data loaded with {len(pairs)} pairs.')

    def _load_netcdf(self, path):
        if not path.exists():
            raise FileNotFoundError(f"NetCDF file not found at {path}")
        return xr.open_dataset(path, decode_times=False)

    def _load_pickle(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def _save_netcdf(self, data, path):
        data.to_netcdf(path)
        print(f"Data saved to {path}")

    def _save_pickle(self, data, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def _create_interim(self):
        # Replace with your actual logic
        print("Processing raw data to interim.")
        raw_data = self._load_netcdf(self.raw_path)

        interim_data = self._apply_subsetting(raw_data, self.arguments)

        # remove raw_data
        del raw_data

        return interim_data

    def _create_model_input(self):
        # Replace with your actual logic
        print("Transforming interim data to feature label pairs.")

        pairs = self._make_feature_label_pairs(self.interim_path, self.arguments)

        return pairs
    
    def _apply_subsetting(self, raw_data, args):
        """
        Apply subsetting to the raw data.
        """

        print("Applying processing to raw data.")

        interim_data = raw_data[args['inputs']]

        # not yet implemented
        # if args['subset_time'] is not None:
        #     raw_data = raw_data.isel(time_counter=slice(args['subset_time'][0], args['subset_time'][1]))
        #     print(f"Subsetting time from {args['subset_time'][0]} to {args['subset_time'][1]}")
        
        if args['subset_region'] == 'Arctic':
            interim_data = interim_data.where(interim_data['nav_lat'] > 60, drop=True)
        elif args['subset_region'] == 'Antarctic':
            interim_data = interim_data.where(interim_data['nav_lat'] < -60, drop=True)
        print(f"Subsetting region to {args['subset_region']}")

        return interim_data
    
    def _make_feature_label_pairs(self, interim_path, args):

        data = self._load_netcdf(interim_path)
        print(f"Making feature-label pairs from {interim_path}")

        # Get rid of data where siconc=0
        data = data.where(data.siconc > 0, drop=True)
    
        # Flatten data
        data = data.stack(xy=('x', 'y'))
        data = data.dropna(dim='xy')

        features = data[args['features']]
        labels = data[args['labels']]
        xr_pairs = []
        for itime in range(features.time_counter.size - 1):
            feature = features.isel(time_counter=itime)
            label = labels.isel(time_counter=itime+1)
            xr_pairs.append((feature, label))

        # Convert xr.datasets to pandas DataFrames
        pairs = []
        for pair in xr_pairs:
            feature = pair[0].to_dataframe()[args['features']].values
            label = pair[1].to_dataframe()[args['labels']].values
            pairs.append((feature, label))

        return pairs
    


    

    
