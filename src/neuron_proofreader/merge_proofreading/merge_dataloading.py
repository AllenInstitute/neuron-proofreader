"""
Created on Thu Nov 13 11:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for loading merge site dataset.

"""

import ast
import numpy as np
import pandas as pd


# --- Load Skeletons ---


# --- Process Merge Site DataFrame ---
def get_brain_segmentation_pairs(merge_sites_df):
    """
    Extracts unique (brain_id, segmentation_id) pairs from a merge sites
    dataframe.

    Parameters
    ----------
    merge_sites_df : pandas.DataFrame
        DataFrame containing merge site information. Must have columns:
            - 'brain_id' : unique identifier of the whole-brain dataset
            - 'segmentation_id' : unique identifier of the segmentation

    Returns
    -------
    brain_segmentation_pairs : Set[Tuple[str]]
        Unique (brain_id, segmentation_id) pairs from a merge sites dataframe.
    """
    brain_segmentation_pairs = set()
    for i in range(len(merge_sites_df)):
        brain_id = merge_sites_df["brain_id"][i]
        segmentation_id = merge_sites_df["segmentation_id"][i]
        brain_segmentation_pairs.add((brain_id, segmentation_id))
    return brain_segmentation_pairs


def get_brain_merge_sites(merge_sites_df, brain_id):
    """
    Gets the xyz coordinates of ground truth merge sites for a given brain.

    Parameters
    ----------
    merge_sites_df : pandas.DataFrame
        DataFrame containing merge sites, must contain the columns:
        "brain_id", "segmentation_id", "segment_id", and "xyz".
    brain_id : str
        Unique identifier for a whole-brain dataset.

    Returns
    -------
    numpy.ndarray
        Ground-truth merge sites (xyz coordinates) for a given brain.
    """
    idx_mask = merge_sites_df["brain_id"] == brain_id
    return np.array(merge_sites_df.loc[idx_mask, "xyz"].tolist())


def load_merge_sites_df(path):
    """
    Loads a merge sites dataframe from a CSV file and process its columns.

    Parameters
    ----------
    path : str
        Path to the CSV file containing merge site data. The CSV must include
        the columns: 'brain_id', 'segment_id', and 'xyz'.

    Returns
    -------
    merge_sites_df : pandas.DataFrame
        Processed dataframe with the following modifications:
            - 'brain_id' and 'segment_id' converted to strings.
            - 'xyz' converted from string representation to tuple
    """
    merge_sites_df = pd.read_csv(path)
    merge_sites_df["brain_id"] = merge_sites_df["brain_id"].apply(str)
    merge_sites_df["segment_id"] = merge_sites_df["segment_id"].apply(str)
    merge_sites_df["xyz"] = merge_sites_df["xyz"].apply(ast.literal_eval)
    return merge_sites_df
