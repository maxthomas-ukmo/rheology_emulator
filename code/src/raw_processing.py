import xarray as xr
import os
import dask
import numpy as np

# TODO: populate this class such that data for training the model, evaluating the model, and eventaully plotting geospatial data
class RheologyData(argumens):
    pass

def load_raw_data(input_path):
    """
    Load raw data from the specified input path.
    """

    print(f"Loading raw data from {input_path}")

    return xr.open_dataset(input_path, decode_times=False)

def apply_subsetting(raw_data, args):
    """
    Apply subsetting to the raw data.
    This is a placeholder function; actual processing logic should be implemented here.
    """

    print("Applying processing to raw data...")

    raw_data = raw_data[args['inputs']]

    # not yet implemented
    # if args['subset_time'] is not None:
    #     raw_data = raw_data.isel(time_counter=slice(args['subset_time'][0], args['subset_time'][1]))
    #     print(f"Subsetting time from {args['subset_time'][0]} to {args['subset_time'][1]}")
    
    if args['subset_region'] == 'Arctic':
        raw_data = raw_data.where(raw_data['nav_lat'] > 60, drop=True)
    elif args['subset_region'] == 'Antarctic':
        raw_data = raw_data.where(raw_data['nav_lat'] < -60, drop=True)
    print(f"Subsetting region to {args['subset_region']}")


    return raw_data

def process_save_intermediate_data(input_path, output_path, args):
    """
    Save the processed data to the specified output path.
    """

    print(f"Loading raw data from {input_path}")

    raw_data = load_raw_data(input_path)

    print(f"Processing raw data...")

    intermediate_data = apply_subsetting(raw_data, args)

    print(f"Saving processed data to {output_path}")

    intermediate_data.to_netcdf(output_path)

    # intermediate_data = intermediate_data.chunk({'time_counter': -1})
    
    # # Save the processed data to a NetCDF file
    # intermediate_data.to_netcdf(output_path, compute=False)
    
    # print(np.nanmax(intermediate_data['siconc']))
    # # Trigger computation
    # dask.compute(intermediate_data)
    # print(np.nanmax(intermediate_data['siconc']))

    print("Processed data saved successfully.")

def make_feature_label_pairs(input_path, args):

    data = xr.open_dataset(input_path)
    print(f"Making feature-label pairs from {input_path}")

    # Get rid of data where siconc=0
    data = data.where(data.siconc > 0, drop=True)
 
    # Flatten data
    data = data.stack(xy=('x', 'y'))
    data = data.dropna(dim='xy')

    features = data[args['features']]
    labels = data[args['labels']]
    pairs = []
    for itime in range(features.time_counter.size - 1):
        feature = features.isel(time_counter=itime)
        label = labels.isel(time_counter=itime+1)
        pairs.append((feature, label))
    return pairs

def save_feature_label_pairs(pairs, output_path):
    """
    Save the feature-label pairs to the specified output path.
    """

    print(f"Saving feature-label pairs to {output_path}")

    with open(output_path, 'wb') as f:
        import pickle
        pickle.dump(pairs, f)

    print("Feature-label pairs saved successfully.")

def process_save_pairs(input_path, output_path, args):
    """
    Process the raw data and save the feature-label pairs.
    """

    pairs = make_feature_label_pairs(input_path, args)

    save_feature_label_pairs(pairs, output_path)








