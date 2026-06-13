"""
Created on Wed June 5 16:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for working with SWC files. An SWC file is a text-based file format
used to represent the directed graphical structure of a neuron. It contains a
series of nodes such that each has the following attributes:
    "id" (int): node ID
    "type" (int): node type (e.g. soma, axon, dendrite)
    "x" (float): x coordinate
    "y" (float): y coordinate
    "z" (float): z coordinate
    "pid" (int): node ID of parent

Note: Each uncommented line in an SWC file corresponds to a node and contains
      these attributes in the same order.
"""

from botocore import UNSIGNED
from botocore.config import Config
from collections import deque
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from google.auth.exceptions import TransportError
from google.cloud import storage
from io import BytesIO, StringIO
from tqdm import tqdm
from zipfile import ZipFile

import ast
import boto3
import networkx as nx
import numpy as np
import os

from neuron_proofreader.utils import util


class Reader:
    """
    Class that reads SWC files stored in a (1) local directory, (2) local ZIP
    archive, and (3) local directory of ZIP archives.
    """

    def __init__(
        self, anisotropy=(1.0, 1.0, 1.0), min_swc_pts=1, verbose=True
    ):
        """
        Instantiates a Reader object for reading SWC files.

        Parameters
        ----------
        anisotropy : Tuple[float], optional
            Image to physical coordinates scaling factors to account for the
            anisotropy of the microscope. Default is (1.0, 1.0, 1.0).
        min_swc_pts : int, optional
            ...
        verbose : bool, optional
            Indication of whether to display a progress bar. Default is True.
        """
        self.anisotropy = anisotropy
        self.min_swc_pts = min_swc_pts
        self.verbose = verbose

    # --- Read Data ---
    def __call__(self, swc_pointer):
        """
        Loads SWC files based on the type pointer provided.

        Parameters
        ----------
        swc_pointer : str
            Object that points to SWC files to be read, must be one of:
                - file_path: Path to single SWC file
                - dir_path: Path to local directory with SWC files
                - zip_path: Path to local ZIP with SWC files
                - zip_dir_path: Path to local directory of ZIPs with SWC files
                - s3_dir_path: Path to S3 prefix with SWC files
                - gcs_dir_path: Path to GCS prefix with SWC files
                - gcs_zip_dir_path: Path to GCS prefix with ZIPs of SWC files

        Returns
        -------
        Deque[dict]
            Dictionaries whose keys and values are the attribute names and
            values from the SWC files. Each dictionary contains the following:
            items:
                - "id": unique identifier of each node in an SWC file.
                - "pid": parent ID of each node.
                - "radius": radius value corresponding to each node.
                - "xyz": coordinate corresponding to each node.
                - "filename": filename of SWC file
                - "swc_id": name of SWC file, minus the ".swc".
        """
        # List of paths
        if isinstance(swc_pointer, list):
            return self.read_swcs(swc_pointer)

        # Directory containing...
        if os.path.isdir(swc_pointer):
            # Local ZIP archives with SWC files
            paths = util.list_paths(swc_pointer, extension=".zip")
            if len(paths) > 0:
                return self.read_zips(swc_pointer, self.read_zip)

            # Local SWC files
            paths = util.read_paths(swc_pointer, extension=".swc")
            if len(paths) > 0:
                return self.read_swcs(paths)

            raise Exception("Directory is Invalid!")

        # Path to...
        if isinstance(swc_pointer, str):
            # Cloud GCS/S3 storage
            if util.is_gcs_path(swc_pointer) or util.is_s3_path(swc_pointer):
                return self.read_from_cloud(swc_pointer)

            # Local ZIP archive with SWC files
            if swc_pointer.endswith(".zip"):
                return self.read_zip(swc_pointer)

            # Local path to single SWC file
            if swc_pointer.endswith(".swc"):
                return self.read_swc(swc_pointer)

            raise Exception("Path is Invalid!")

        raise Exception("SWC Pointer is Invalid!")

    def read_swc(self, path):
        """
        Reads a single SWC file.

        Paramters
        ---------
        path : str
            Path to SWC file.

        Returns
        -------
        dict
            Dictionary whose keys and values are the attribute names and
            values from an SWC file.
        """
        content = util.read_txt(path).splitlines()
        filename = os.path.basename(path)
        return self.parse(content, filename)

    def read_swcs(self, swc_paths):
        """
        Reads SWC files stored in a GCS or S3 bucket.

        Parameters
        ----------
        swc_paths : List[str]
            List of paths to SWC files to be read.

        Returns
        -------
        swc_dicts : Deque[dict]
            Dictionaries whose keys and values are the attribute names and
            values from an SWC file.
        """
        with ThreadPoolExecutor() as executor:
            # Assign threads
            threads = {
                executor.submit(self.read_swc, p) for p in swc_paths[0:1]
            }
            pbar = self.manual_progress_bar(len(threads))

            # Store results
            swc_dicts = deque()
            for thread in as_completed(threads):
                result = thread.result()
                if result:
                    swc_dicts.append(result)
                if self.verbose:
                    pbar.update(1)
        return swc_dicts

    def read_zips(self, zip_paths, read_fn):
        """
        Reads SWC files stored in ZIP archives.

        Parameters
        ----------
        bucket_name : str
            Name of bucket containing SWC files.
        zip_paths : List[str]
            Paths to ZIP archives containing SWC files to be read.

        Returns
        -------
        swc_dicts : Deque[dict]
            Dictionaries whose keys and values are the attribute names and
            values from an SWC file.
        """
        pbar = self.manual_progress_bar(len(zip_paths))
        with ProcessPoolExecutor() as executor:
            # Assign processes
            futures = {executor.submit(read_fn, path) for path in zip_paths}

            # Store results
            swc_dicts = deque()
            for process in as_completed(futures):
                try:
                    swc_dicts.extend(process.result())
                except Exception:
                    pass

                if self.verbose:
                    pbar.update(1)
        return swc_dicts

    def read_zip(self, zip_path):
        """
        Reads SWC files from a ZIP archive.

        Paramters
        ---------
        zip_path : str
            Path to ZIP archive.

        Returns
        -------
        swc_dicts : Deque[dict]
            Dictionaries whose keys and values are the attribute names and
            values from an SWC file.
        """
        with ThreadPoolExecutor() as executor:
            # Assign threads
            threads = set()
            zf = ZipFile(zip_path, "r")
            for name in [f for f in zf.namelist() if f.endswith(".swc")]:
                threads.add(executor.submit(self.read_zipped_swc, zf, name))

            # Store results
            swc_dicts = deque()
            for thread in as_completed(threads):
                result = thread.result()
                if result:
                    swc_dicts.append(result)
        return swc_dicts

    def read_zipped_swc(self, zipfile, path):
        """
        Reads an SWC file stored in a ZIP archive.

        Parameters
        ----------
        zipfile : ZipFile
            ZIP archive containing SWC files.
        path : str
            Path to SWC file.

        Returns
        -------
        dict
            Dictionary whose keys and values are the attribute names and
            values from an SWC file.
        """
        content = util.read_zip(zipfile, path).splitlines()
        filename = os.path.basename(path)
        return self.parse(content, filename)

    def read_from_cloud(self, path):
        """
        Reads SWC files stored in a GCS or S3 bucket.

        Parameters
        ----------
        path : str
            Path to location in a GCS or S3 bucket containing SWC files,
            must be in the format "{scheme}://{bucket_name}/{prefix}".

        Returns
        -------
        Deque[dict]
            Dictionaries whose keys and values are the attribute names and
            values from an SWC file.
        """
        # Extract info
        assert util.is_s3_path(path) or util.is_gcs_path(path)
        use_s3 = util.is_s3_path(path)

        # List paths
        swc_paths = util.list_cloud_paths(path, ".swc")
        zip_paths = util.list_cloud_paths(path, ".zip")

        # Call reader
        if swc_paths:
            return self.read_swcs(swc_paths)
        elif zip_paths:
            read_fn = self.read_s3_zip if use_s3 else self.read_gcs_zip
            return self.read_zips(zip_paths, read_fn)
        else:
            return list()

    def read_gcs_swc(self, path):
        """
        Reads a single SWC file stored in a GCS bucket.

        Parameters
        ----------
        path : List[str]
            Path to SWC file to be read.

        Returns
        -------
        Deque[dict]
            Dictionaries whose keys and values are the attribute names and
            values from an SWC file.
        """
        # Initialize cloud reader
        bucket_name, key = util.parse_cloud_path(path)
        bucket = storage.Client().bucket(bucket_name)
        blob = bucket.blob(key)

        # Parse swc contents
        content = blob.download_as_text().splitlines()
        filename = os.path.basename(key)
        return self.parse(content, filename)

    def read_gcs_zip(self, path):
        """
        Reads SWC files stored in a ZIP archive downloaded from a GCS
        bucket.

        Parameters
        ----------
        path : str
            Path to ZIP archive containing SWC files to be read.

        Returns
        -------
        swc_dicts : Deque[dict]
            Dictionaries whose keys and values are the attribute names and
            values from an SWC file.
        """
        bucket_name, key = util.parse_cloud_path(path)
        bucket = storage.Client().bucket(bucket_name)
        try:
            zip_content = bucket.blob(key).download_as_bytes()
        except TransportError:
            print(f"Failed to read {path}!")
            return deque()
        return self._parse_zip_bytes(zip_content)

    def read_s3_zip(self, path):
        """
        Reads SWC files stored in a ZIP archive downloaded from an S3
        bucket.

        Parameters
        ----------
        path : str
            Path to ZIP archive containing SWC files to be read.

        Returns
        -------
        swc_dicts : Deque[dict]
            Dictionaries whose keys and values are the attribute names and
            values from an SWC file.
        """
        bucket, key = util.parse_cloud_path(path)
        s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        zip_content = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        return self._parse_zip_bytes(zip_content)

    def _parse_zip_bytes(self, zip_content):
        with ZipFile(BytesIO(zip_content), "r") as zf:
            names = [f for f in zf.namelist() if f.endswith(".swc")]
            with ThreadPoolExecutor() as executor:
                threads = {
                    executor.submit(self.read_zipped_swc, zf, name)
                    for name in names
                }
                return deque(
                    t.result() for t in as_completed(threads) if t.result()
                )

    # -- Process Text ---
    def manual_progress_bar(self, total):
        """
        Gets progress bar that needs to be updated manually.

        Parameters
        ----------
        total : int
            Size of progress bar.

        Returns
        -------
        tqdm.tqdm
            Iterator that is optionally wrapped in a progress bar.
        """
        return tqdm(total=total, desc="Read SWCs") if self.verbose else None

    def parse(self, content, filename):
        """
        Parses an SWC file to extract the content which is stored in a dict.

        Parameters
        ----------
        content : List[str]
            List of strings such that each is a line from an SWC file.

        Returns
        -------
        dict
            Dictionary whose keys and values are the attribute names and
            values from an SWC file.
        """
        # Initializations
        swc_name, _ = os.path.splitext(filename)
        content, offset = self.process_content(content)
        if len(content) >= self.min_swc_pts:
            swc_dict = {
                "id": np.zeros((len(content)), dtype=int),
                "pid": np.zeros((len(content)), dtype=int),
                "radius": np.zeros((len(content)), dtype=float),
                "xyz": np.zeros((len(content), 3), dtype=np.int32),
                "soma_nodes": set(),
                "swc_name": swc_name,
            }

            # Parse content
            for i, line in enumerate(content):
                parts = line.split()
                swc_dict["id"][i] = parts[0]
                swc_dict["pid"][i] = parts[-1]
                swc_dict["radius"][i] = float(parts[-2])
                swc_dict["xyz"][i] = self.read_coordinate(parts[2:5], offset)

                if int(parts[1]) == 1:
                    swc_dict["soma_nodes"].add(parts[0])

            # Convert radius from nanometers to microns
            if swc_dict["radius"][0] > 100:
                swc_dict["radius"] /= 1000
            return swc_dict
        else:
            return None

    def process_content(self, content):
        """
        Processes lines of text from an SWC file, extracting an offset
        value and returning the remaining content starting from the line
        immediately after the last commented line.

        Parameters
        ----------
        content : List[str]
            List of strings such that each is a line from an SWC file.

        Returns
        -------
        content : List[str]
            Lines from an SWC file after comments.
        offset : Tuple[int]
            Offset used to shift coordinate.
        """
        offset = (0, 0, 0)
        for i, line in enumerate(content):
            if line.startswith("# OFFSET"):
                parts = line.split()
                offset = self.read_coordinate(parts[2:5])
            if not line.startswith("#") and len(line.strip()) > 0:
                return content[i:], offset
        return [], offset

    def read_coordinate(self, xyz_str, offset=(0, 0, 0)):
        """
        Reads a coordinate from a string and converts it to voxel coordinates.

        Parameters
        ----------
        xyz_str : str
            Coordinate stored as a string.
        offset : Tuple[int]
            Offset of coordinates in SWC file. Default is (0, 0, 0).

        Returns
        -------
        Tuple[int]
            xyz coordinates of an entry from an SWC file.
        """
        iterator = zip(self.anisotropy, xyz_str, offset)
        return [a * (float(s) + o) for a, s, o in iterator]


# --- Write ---
def write_points(
    zip_path, points, color=None, prefix="", radius=10, write_mode="w"
):
    """
    Writes a list of 3D points to individual SWC files in the specified
    directory.

    Parameters
    -----------
    zip_path : str
        Path to ZIP archive where the SWC files will be saved.
    points : List[Tuple[float]]
        List of 3D points to be saved.
    color : str, optional
        Color to associate with the points in the SWC files. Default is
        None.
    prefix : str, optional
        String that is prefixed to the filenames of the SWC files. Default is
        an empty string. Default is an empty string.
    radius : float, optional
        Radius to be used in SWC file. Default is 10.
    """
    zf = ZipFile(zip_path, write_mode)
    for i, xyz in enumerate(points):
        filename = prefix + str(i + 1) + ".swc"
        to_zipped_point(zf, filename, xyz, color=color, radius=radius)


def to_zipped_point(zf, filename, xyz, color=None, radius=5):
    """
    Writes a point to an SWC file format, which is then stored in a ZIP
    archive.

    Parameters
    ----------
    zf : zipfile.ZipFile
        ZipFile used to write the generated SWC file.
    filename : str
        Filename of SWC file.
    xyz : ArrayLike
        Point to be written to SWC file.
    color : str, optional
        Color of nodes. Default is None.
    radius : float, optional
        Radius (in microns) of point. Default is 5.
    """
    with StringIO() as text_buffer:
        # Preamble
        if color:
            text_buffer.write("# COLOR " + color)
        text_buffer.write("\n" + "# id, type, z, y, x, r, pid")

        # Write entry
        x, y, z = tuple(xyz)
        text_buffer.write("\n" + f"1 5 {x} {y} {z} {radius} -1")

        # Finish
        zf.writestr(filename, text_buffer.getvalue())


# --- Helpers ---
def get_segment_id(swc_name):
    """
    Extract the segment ID from an SWC filename.

    Parameters
    ----------
    swc_name : str
        SWC filename in the format "{segment_id}.swc".

    Returns
    -------
    int or str
        Segment ID parsed as an integer if possible; otherwise, the original
        string.
    """
    try:
        return ast.literal_eval(swc_name.split(".")[0])
    except:
        return swc_name


def get_swc_name(path):
    """
    Gets name of the SWC file at the given path, minus the extension.

    Parameters
    ----------
    path : str
        Path to SWC file.

    Returns
    -------
    name : str
        Name of the SWC file, minus the extension.
    """
    return os.path.splitext(os.path.basename(path))[0]


def to_graph(swc_dict):
    """
    Converts an SWC dictionary to a NetworkX graph with reindexed nodes.

    Parameters
    ----------
    swc_dict : dict
        Contents of an SWC file.
    set_attrs : bool, optional
        Indication of whether to set "xyz" and "radius" as graph-level
        attributes. Default is False.

    Returns
    -------
    graph : networkx.Graph
        Graph built from an SWC file.
    """
    # Reindex nodes: map swc ids to 0...N-1
    id_map = {old_id: new_id for new_id, old_id in enumerate(swc_dict["id"])}
    edges = [
        (id_map[child], id_map[parent])
        for child, parent in zip(swc_dict["id"][1:], swc_dict["pid"][1:])
    ]

    # Build graph with reindexed edges
    graph = nx.Graph(
        swc_name=swc_dict["swc_name"],
        radius=swc_dict["radius"],
        xyz=swc_dict["xyz"],
    )
    graph.add_edges_from(edges)
    return graph
