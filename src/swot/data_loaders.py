#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file contains routines for opening locally-saved SWOT (Surface Water and Ocean Topography) data.

Functions:
1. remap_quality_flags: Reassigns discrete quality flags in the data to a smaller, simpler range.
2. load_cycle: Loads and processes SWOT data for a specified cycle and optionally filters it.

Author: Tatsu, comments partially by ChatGPT :,)
Date: First version: 1.23.2025

Dependencies:
    - xarray
    - numpy
"""

import os
import xarray as xr
import s3fs
import swot.swot_utils as swot_utils  # Custom utilities for SWOT data


def _get_fs(path: str):
    if path.startswith("s3://"):
        return s3fs.S3FileSystem(anon=True), True
    return None, False

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Function: remap_quality_flags
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def remap_quality_flags(swath):
    """
    Remaps quality flags in the SWOT dataset to simplified numeric categories.

    Parameters
    ----------
    swath : xarray.Dataset
        The SWOT dataset containing the 'quality_flag' variable.

    Returns
    -------
    xarray.Dataset
        The modified dataset with remapped quality_flag values.

    Notes
    -----
    - The function maps the following original flag values to a new range:
        5   -> 1
        10  -> 2
        20  -> 3
        30  -> 4
        50  -> 5
        70  -> 6
        100 -> 7
        101 -> 8
        102 -> 9
    - This simplifies plotting and interpretation of quality flags.
    """
    # Check if the 'quality_flag' variable exists in the dataset
    if not "quality_flag" in swath:
        return

    # Access the 'quality_flag' variable
    flags = swath.quality_flag

    # Remap the quality flag values using direct value replacement
    flags.values[flags.values == 5.] = 1
    flags.values[flags.values == 10.] = 2
    flags.values[flags.values == 20.] = 3
    flags.values[flags.values == 30.] = 4
    flags.values[flags.values == 50.] = 5
    flags.values[flags.values == 70.] = 6
    flags.values[flags.values == 100.] = 7
    flags.values[flags.values == 101.] = 8
    flags.values[flags.values == 102.] = 9

    # Update the 'quality_flag' variable in the dataset
    swath.quality_flag.values = flags.values
    
    return swath


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Function: load_cycle
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def load_cycle(path, cycle="002", pass_ids=None, fields=None, subset=False, lats=[-90, 90]):
    """
    Loads SWOT data for a specific cycle from locally stored NetCDF files.

    Parameters
    ----------
    path : str
        Path to the root directory containing SWOT data cycles.
    cycle : str, optional
        The cycle number to load (default is "002").
    pass_ids : list of str, optional
        Specific pass IDs to load. If None, all passes are loaded (default: None).
    fields : list of str, optional
        Fields (variables) to extract from the dataset. If None, all fields are loaded (default: None).
    subset : bool, optional
        Whether to subset the data spatially by latitude (default: False).
    lats : list of float, optional
        Latitude range [min_lat, max_lat] to use for subsetting if `subset` is True (default: [-90, 90]).

    Returns
    -------
    list of xarray.Dataset
        A list of datasets, one for each loaded SWOT pass.

    Notes
    -----
    - Datasets are filtered by cycle and optionally by pass ID.
    - Remaps 'quality_flag' values for easier discrete plotting.
    """
    fs, is_s3 = _get_fs(path)
    cycle_dir = f"{path}/cycle_{cycle}"

    if is_s3:
        cycle_dir_s3 = cycle_dir.replace("s3://", "")
        if not fs.exists(cycle_dir_s3):
            print(f"Can't find path {cycle_dir}")
            return []
        all_files = [f.split("/")[-1] for f in fs.ls(cycle_dir_s3)]
    else:
        if not os.path.exists(cycle_dir):
            print(f"Can't find path {cycle_dir}")
            return []
        all_files = os.listdir(cycle_dir)

    if pass_ids is None:
        swot_passes = [f for f in all_files if ".nc" in f]
    else:
        swot_passes = []
        for pass_id in pass_ids:
            swot_passes += [f for f in all_files if f"{cycle}_{pass_id}" in f]

    swot_passes = sorted(swot_passes, key=lambda x: int(x.split("_")[6]))

    passes = []

    for swot_pass in swot_passes:
        print(f"Loading {swot_pass}")
        try:
            file_path = f"{cycle_dir}/{swot_pass}"
            if is_s3:
                with fs.open(file_path.replace("s3://", "")) as f:
                    swath = xr.open_dataset(f, engine="h5netcdf").load()
            else:
                swath = xr.open_dataset(file_path).load()

            # If specific fields are requested, extract only those
            if fields is None:
                fields = list(swath.variables)  # Load all variables if none are specified
            swath = swath[fields]

            # Subset the data by latitude if requested
            if subset:
                swath = swot_utils.subset(swath, lats)

            # Add cycle and pass ID as metadata attributes
            swath = swath.assign_attrs(
                cycle=f"{cycle}",
                pass_ID=f"{swot_pass.split('_')[6]}"
            )

            # Remap the quality flags for discrete plotting
            if "quality_flag" in fields:
                swath = remap_quality_flags(swath)

            # Append the processed swath to the list
            passes.append(swath)
        except Exception as e:
            print("Whoops, can't open dataset")
            print("An error occurred:", e)

    # Return the list of loaded and processed passes
    return passes
