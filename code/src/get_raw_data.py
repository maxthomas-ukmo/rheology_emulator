# Becuase of MO rules on publishing paths, this script should take a path as an argument to grab and cut down the initial data
import xarray as xr
import sys
import dask

def get_raw_data(get_path, potential_inputs=None):

    if potential_inputs is None:
        potential_inputs = ['sithic', 'sivolu', 'siconc', 'sivpnd', 'sivelu', 'sivelv', 'sivelo', 'utau_ai', 'vtau_ai', 'utau_oi', 'vtau_oi', 'sidive', 'sishea', 'sistre', 'normstr', 'sheastr']

    print(f"Retrieving raw data from {get_path}")

    raw = xr.open_dataset(get_path)

    return raw[potential_inputs]


def save_raw_data(output_path, raw_data):

    print(raw_data)

    raw_data = raw_data.chunk({'time_counter': -1})

    raw_data.to_netcdf(output_path, compute=False)

    dask.compute(raw_data)


def get_and_save(get_path, output_path, potential_inputs=None):
    raw_data = get_raw_data(get_path, potential_inputs)
    save_raw_data(output_path, raw_data)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python get_training_data.py <get_path> <output_path> [potential_inputs]")
        sys.exit(1)

    get_path = sys.argv[1]
    output_path = sys.argv[2]
    potential_inputs = sys.argv[3:] if len(sys.argv) > 3 else None

    get_and_save(get_path, output_path, potential_inputs)
