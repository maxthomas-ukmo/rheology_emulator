# Becuase of MO rules on publishing paths, this script should take a path as an argument to grab and cut down the initial data
import xarray as xr
import sys
import dask
import numpy as np

def get_raw_data(get_path, inputs=None):

    if inputs is None:
        inputs = ['sithic', 'sivolu', 'siconc', 'sivpnd', 'sivelu', 'sivelv', 'sivelo', 'utau_ai', 'vtau_ai', 'utau_oi', 'vtau_oi', 'sidive', 'sishea', 'sistre', 'normstr', 'sheastr']

    print(f"Retrieving raw data from {get_path}")

    raw = xr.open_dataset(get_path)

    print(np.nanmax(raw['siconc']))

    return raw[inputs]


def save_raw_data(output_path, raw_data):

    print(raw_data)

    raw_data = raw_data.chunk({'time_counter': -1})

    print(np.nanmax(raw_data['siconc']))

    raw_data.to_netcdf(output_path, compute=False)

    print(np.nanmax(raw_data['siconc']))

    print('NOTE: This isnt working yet. Probably all of the raw getting needs to be submitted with qsub and loads of memory.')
    
    dask.compute(raw_data)


def get_and_save(get_path, output_path, inputs=None):
    raw_data = get_raw_data(get_path, inputs)
    save_raw_data(output_path, raw_data)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python get_training_data.py <get_path> <output_path> [inputs]")
        sys.exit(1)

    get_path = sys.argv[1]
    output_path = sys.argv[2]
    inputs = sys.argv[3:] if len(sys.argv) > 3 else None

    get_and_save(get_path, output_path, inputs)
