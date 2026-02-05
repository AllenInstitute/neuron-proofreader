"""
Created on Thu Nov 13 11:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for loading merge site dataset.

"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import ast
import numpy as np
import os
import pandas as pd

from neuron_proofreader.utils import util

TEST_BRAIN = "653159"


# --- Load Skeletons ---
def load_fragments(dataset, is_test=False, max_workers=2):
    """
    Loads neuron fragments for a selected set of merge-site indices into
    dataset.

    Parameters
    ----------
    dataset : MergeSiteDataset
        Dataset that fragments are loaded into.
    is_test : bool, optional
        Indication of whether this is a test run so only fragments from a
        single brain should be loaded. Default is False.
    max_workers : int, optional
        Maximum number of parallel workers. Default is 8.
    """
    # Initializations
    merge_sites_df = dataset.merge_sites_df
    target_pairs = get_brain_segmentation_pairs(merge_sites_df)
    root = "gs://allen-nd-goog/automated_proofreading_dataset/raw_merge_sites"

    # Build list of (brain_id, swc_pointer) pairs to load
    load_tasks = []
    for brain_id in get_brain_ids(merge_sites_df, is_test):
        sub_df = merge_sites_df.loc[merge_sites_df["brain_id"] == brain_id]
        for segmentation_id in sub_df["segmentation_id"].unique():
            if (brain_id, segmentation_id) in target_pairs:
                swc_pointer = f"{root}/{brain_id}/{segmentation_id}/merged_fragments.zip"
                load_tasks.append((brain_id, swc_pointer))

    # Load in parallel
    print(f"\nLoading Fragments ({len(load_tasks)} brains, {max_workers} workers)")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                dataset.load_fragment_graphs, brain_id, swc_pointer, False
            ): brain_id
            for brain_id, swc_pointer in load_tasks
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fragments"):
            brain_id = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error loading fragments for {brain_id}: {e}")


def load_groundtruth(dataset, is_test=False, max_workers=2):
    """
    Loads ground truth skeletons into dataset.

    Parameters
    ----------
    dataset : MergeSiteDataset
        Dataset that fragments are loaded into.
    is_test : bool, optional
        Indication of whether this is a test run so only fragments from a
        single brain should be loaded. Default is False.
    max_workers : int, optional
        Maximum number of parallel workers. Default is 8.
    """
    root = "gs://allen-nd-goog/ground_truth_tracings"
    brain_ids = get_brain_ids(dataset.merge_sites_df, is_test)

    print(f"\nLoading Ground Truth ({len(brain_ids)} brains, {max_workers} workers)")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                dataset.load_gt_graphs, brain_id, f"{root}/{brain_id}/voxel"
            ): brain_id
            for brain_id in brain_ids
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Ground Truth"):
            brain_id = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error loading ground truth for {brain_id}: {e}")


def load_images(
    dataset, img_prefixes_path, segmentation_prefixes_path, is_test=False, max_workers=2
):
    """
    Loads images into dataset.

    Parameters
    ----------
    dataset : MergeSiteDataset
        Dataset that fragments are loaded into.
    img_prefixes_path : str, optional
        Path to json that maps brain IDs to S3 image paths. Default is None.
    segmentation_prefixes_path : str, optional
        Path to json that maps brain IDs to segmentation paths. Default is
        None.
    is_test : bool, optional
        Indication of whether this is a test run so only fragments from a
        single brain should be loaded. Default is False.
    max_workers : int, optional
        Maximum number of parallel workers. Default is 8.
    """
    img_prefixes = util.read_json(img_prefixes_path)
    segmentation_prefixes = util.read_json(segmentation_prefixes_path)
    brain_ids = get_brain_ids(dataset.merge_sites_df, is_test)

    print(f"\nLoading Images ({len(brain_ids)} brains, {max_workers} workers)")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                dataset.load_images,
                brain_id,
                os.path.join(img_prefixes[brain_id], "0"),
                segmentation_prefixes[brain_id]
            ): brain_id
            for brain_id in brain_ids
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Images"):
            brain_id = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error loading images for {brain_id}: {e}")


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
    pairs = set()
    for i in range(len(merge_sites_df)):
        brain_id = merge_sites_df["brain_id"][i]
        segmentation_id = merge_sites_df["segmentation_id"][i]
        pairs.add((brain_id, segmentation_id))
    return pairs


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


def load_merge_sites_df(path, is_test=False):
    """
    Loads a merge sites dataframe from a CSV file and process its columns.

    Parameters
    ----------
    path : str
        Path to the CSV file containing merge site data. The CSV must include
        the columns: 'brain_id', 'segment_id', and 'xyz'.
    is_test : bool, optional
        Indication of whether this is a test run so only sites from a single
        brain should be loaded. Default is False.

    Returns
    -------
    merge_sites_df : pandas.DataFrame
        Processed dataframe with the following modifications:
            - 'brain_id' and 'segment_id' converted to strings.
            - 'xyz' converted from string representation to tuple
    """
    # Read and process
    merge_sites_df = pd.read_csv(path)
    merge_sites_df["brain_id"] = merge_sites_df["brain_id"].apply(str)
    merge_sites_df["segment_id"] = merge_sites_df["segment_id"].apply(str)
    merge_sites_df["xyz"] = merge_sites_df["xyz"].apply(ast.literal_eval)

    # Check whether test run
    if is_test:
        idx_mask = merge_sites_df["brain_id"] == TEST_BRAIN
        return merge_sites_df[idx_mask].reset_index(drop=True)
    else:
        return merge_sites_df


# --- Helpers ---
def get_brain_ids(merge_sites_df, is_test=False):
    """
    Gets brain IDs of datasets to be loaded.

    Parameters
    ----------
    merge_sites_df : pandas.DataFrame
        DataFrame containing merge sites, must contain the columns:
        "brain_id", "segmentation_id", "segment_id", and "xyz".
    is_test : bool, optional
        Indication of whether this is a test run so only fragments from a
        single brain should be loaded. Default is False.

    Returns
    -------
    List[str]
        Brain IDs of datasests to be loaded.
    """
    return [TEST_BRAIN] if is_test else merge_sites_df["brain_id"].unique()


def read_idxs(path):
    """
    Reads a list of indices from a CSV file.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    List[int]
        Indices extracted from the CSV file.
    """
    return pd.read_csv(path)["Indices"]
