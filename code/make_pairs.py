import xarray as xr
import numpy as np

# --- User settings ---
features = ['siconc', 'sivelv', 'sivelu']  # replace with your feature variable names
labels = ['sivelv']                        # replace with your label variable names
filelist = [
    "/data/scratch/max.thomas/rheology_training_data/12_day_test_set/dq557o_1ts_19760101_19760101_icemod_19760101-19760101.nc",
    "/data/scratch/max.thomas/rheology_training_data/12_day_test_set/dq557o_1ts_19760201_19760201_icemod_19760201-19760201.nc",
    "/data/scratch/max.thomas/rheology_training_data/12_day_test_set/dq557o_1ts_19760301_19760301_icemod_19760301-19760301.nc",
    "/data/scratch/max.thomas/rheology_training_data/12_day_test_set/dq557o_1ts_19760401_19760401_icemod_19760401-19760401.nc",
    "/data/scratch/max.thomas/rheology_training_data/12_day_test_set/dq557o_1ts_19760501_19760501_icemod_19760501-19760501.nc",
    "/data/scratch/max.thomas/rheology_training_data/12_day_test_set/dq557o_1ts_19760601_19760601_icemod_19760601-19760601.nc",
    "/data/scratch/max.thomas/rheology_training_data/12_day_test_set/dq557o_1ts_19760701_19760701_icemod_19760701-19760701.nc",
    "/data/scratch/max.thomas/rheology_training_data/12_day_test_set/dq557o_1ts_19760801_19760801_icemod_19760801-19760801.nc",
    "/data/scratch/max.thomas/rheology_training_data/12_day_test_set/dq557o_1ts_19760901_19760901_icemod_19760901-19760901.nc",
    "/data/scratch/max.thomas/rheology_training_data/12_day_test_set/dq557o_1ts_19761001_19761001_icemod_19761001-19761001.nc",
    "/data/scratch/max.thomas/rheology_training_data/12_day_test_set/dq557o_1ts_19761101_19761101_icemod_19761101-19761101.nc",
    "/data/scratch/max.thomas/rheology_training_data/12_day_test_set/dq557o_1ts_19761201_19761201_icemod_19761201-19761201.nc"
    ]
output_path = "/data/users/max.thomas/paired_dataset.zarr"

pair_ds_list = []

for input_path in filelist:
    ds = xr.open_dataset(input_path, chunks={'time_counter': 1})
    n_pairs = len(ds.time_counter) - 1

    features_arr = xr.concat(
        [ds[var].isel(time_counter=slice(0, -1)) for var in features], dim="feature"
    ).transpose("time_counter", "feature", "y", "x").data  # Dask array

    labels_arr = xr.concat(
        [ds[var].isel(time_counter=slice(1, None)) for var in labels], dim="label"
    ).transpose("time_counter", "label", "y", "x").data  # Dask array

    pair_ds = xr.Dataset(
        data_vars={
            "features": (["pair", "feature", "y", "x"], features_arr),
            "labels": (["pair", "label", "y", "x"], labels_arr),
        },
        coords={
            "pair": np.arange(n_pairs),
            "feature": features,
            "label": labels,
            "y": ds.y.values,
            "x": ds.x.values,
            "lat": (["y", "x"], ds.nav_lat.values),
            "lon": (["y", "x"], ds.nav_lon.values),
            "time_start": (["pair"], ds.time_counter.values[:-1]),
            "time_end": (["pair"], ds.time_counter.values[1:]),
        }
    )
    pair_ds_list.append(pair_ds)

# Concatenate along the 'pair' dimension if more than one file
if len(pair_ds_list) > 1:
    paired_ds = xr.concat(pair_ds_list, dim="pair")
else:
    paired_ds = pair_ds_list[0]

print(paired_ds)

# Save to
paired_ds.to_zarr(output_path, mode='w')