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

from collections import deque
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from google.auth.exceptions import RefreshError, TransportError
from google.cloud import storage
from io import BytesIO, StringIO
from tqdm import tqdm
from zipfile import ZipFile

import ast
import networkx as nx
import numpy as np
import os

from neuron_proofreader.utils import util


class Reader:
    """
    Class that reads SWC files stored in a (1) local directory, (2) local ZIP
    archive, or (3) GCS directory, (4) GCS directory of ZIP archives.
    """

    def __init__(self, anisotropy=(1.0, 1.0, 1.0), min_size=0):
        """
        Initializes a Reader object that reads SWC files.

        Parameters
        ----------
        anisotropy : Tuple[float], optional
            Image to physical coordinates scaling factors to account for the
            anisotropy of the microscope. Default is [1.0, 1.0, 1.0].
        min_size : int, optional
            Threshold on the number nodes in SWC files that are parsed and
            returned. Default is 0.
        """
        self.anisotropy = anisotropy
        self.min_size = min_size

    def __call__(self, swc_pointer):
        """
        Reads SWC files located at the path specified by "swc_pointer".

        Parameters
        ----------
        swc_pointer : str or List[str]
            Object that points to SWC files to be read, must be one of:
                - file_path: Path to single SWC file
                - dir_path: Path to local directory with SWC files
                - zip_path: Path to local ZIP with SWC files
                - zip_dir_path: Path to local directory of ZIPs with SWC files
                - gcs_dir_path: Path to GCS prefix with SWC files
                - gcs_zip_dir_path: Path to GCS prefix with ZIPs of SWC files
                - path_list: List of paths to local SWC files

        Returns
        -------
        Deque[dict]
            List of dictionaries whose keys and values are the attribute names
            and values from the SWC files. Each dictionary contains the
            following items:
                - "id": unique identifier of each node in an SWC file.
                - "pid": parent ID of each node.
                - "radius": radius value corresponding to each node.
                - "xyz": coordinate corresponding to each node.
                - "soma_nodes": nodes with soma type.
                - "swc_name": name of SWC file, minus the ".swc".
        """
        # List of paths to SWC files
        if isinstance(swc_pointer, list):
            return self.read_from_paths(swc_pointer)

        # Directory containing...
        if os.path.isdir(swc_pointer):
            # ZIP archives with SWC files
            paths = util.list_paths(swc_pointer, extension=".zip")
            if len(paths) > 0:
                return self.read_from_zips(swc_pointer)

            # SWC files
            paths = util.list_paths(swc_pointer, extension=".swc")
            if len(paths) > 0:
                return self.read_from_paths(paths)

            raise Exception(f"Directory is invalid - {swc_pointer}")

        # Path to...
        if isinstance(swc_pointer, str):
            # Single SWC file in GCS
            if util.is_gcs_path(swc_pointer) and swc_pointer.endswith(".swc"):
                bucket_name, path = util.parse_cloud_path(swc_pointer)
                return [self.read_from_gcs_swc(bucket_name, path)]

            # GCS directory
            if util.is_gcs_path(swc_pointer):
                return self.read_from_gcs(swc_pointer)

            # ZIP archive with SWC files
            if swc_pointer.endswith(".zip"):
                return self.read_from_zip(swc_pointer)

            # Single SWC file
            if swc_pointer.endswith(".swc"):
                return self.read_from_path(swc_pointer)

            raise Exception(f"Path is invalid {swc_pointer}")

        raise Exception(f"SWC Pointer is invalid {swc_pointer}")

    # --- Read subroutines ---
    def read_from_paths(self, swc_paths):
        """
        Reads a list of SWC files stored on the local machine.

        Paramters
        ---------
        swc_paths : List[str]
            Paths to SWC files stored on the local machine.

        Returns
        -------
        swc_dicts : Dequeue[dict]
            List of dictionaries whose keys and values are the attribute
            names and values from an SWC file.
        """
        with ProcessPoolExecutor() as executor:
            # Assign processes
            processes = list()
            for path in swc_paths:
                processes.append(
                    executor.submit(self.read, path)
                )

            # Store results
            swc_dicts = deque()
            for process in as_completed(processes):
                result = process.result()
                if result:
                    swc_dicts.append(result)
        return swc_dicts

    def read_from_path(self, path):
        """
        Reads a single SWC file stored on the local machine.

        Paramters
        ---------
        path : str
            Path to SWC file stored on the local machine.

        Returns
        -------
        swc_dict : dict
            Dictionaries whose keys and values are the attribute names and
            values from an SWC file.
        """
        content = util.read_txt(path)
        if len(content) > self.min_size - 10:
            swc_dict = self.parse(content)
            swc_dict["swc_name"] = get_swc_name(path)
            return swc_dict
        else:
            return False

    def read_from_zips(self, zip_dir):
        """
        Reads a directory containing ZIP archives with SWC files.

        Parameters
        ----------
        zip_dir : str
            Path to directory containing ZIP archives with SWC files.

        Returns
        -------
        swc_dicts : Deque[dict]
            Dictionaries whose keys and values are the attribute names and
            values from an SWC file.
        """
        # Initializations
        zip_names = [f for f in os.listdir(zip_dir) if f.endswith(".zip")]
        pbar = tqdm(total=len(zip_names), desc="Read SWCs")

        # Main
        with ProcessPoolExecutor() as executor:
            # Assign threads
            processes = list()
            for f in zip_names:
                zip_path = os.path.join(zip_dir, f)
                processes.append(
                    executor.submit(self.read_from_zip, zip_path)
                )

            # Store results
            swc_dicts = deque()
            for process in as_completed(processes):
                swc_dicts.extend(process.result())
                pbar.update(1)
        return swc_dicts

    def read_from_zip(self, zip_path):
        """
        Reads SWC files from a ZIP archive stored on the local machine.

        Paramters
        ---------
        str : str
            Path to a ZIP archive on the local machine.

        Returns
        -------
        swc_dicts : Dequeue[dict]
            List of dictionaries whose keys and values are the attribute
            names and values from an SWC file.
        """
        with ThreadPoolExecutor() as executor:
            with ZipFile(zip_path, "r") as zf:
                # Submit threads
                threads = list()
                for f in [f for f in zf.namelist() if f.endswith(".swc")]:
                    threads.append(
                        executor.submit(self.read_from_zipped_file, zf, f)
                    )

                # Store results
                swc_dicts = deque()
                for thread in as_completed(threads):
                    swc_dict = thread.result()
                    if swc_dict:
                        swc_dicts.append(swc_dict)
        return swc_dicts

    def read_from_zipped_file(self, zip_file, path):
        """
        Reads SWC file stored in a ZIP archive.

        Parameters
        ----------
        zip_file : ZipFile
            ZIP archive containing SWC file to be read.
        path : str
            Path to SWC file to be read.

        Returns
        -------
        swc_dict : dict
            Dictionaries whose keys and values are the attribute names and
            values from an SWC file.
        """
        content = util.read_zip(zip_file, path).splitlines()
        if len(content) > self.min_size - 10:
            swc_dict = self.parse(content)
            swc_dict["swc_name"] = get_swc_name(path)
            return swc_dict
        else:
            return False

    def read_from_gcs(self, gcs_path):
        """
        Reads SWC files stored in a GCS bucket.

        Parameters
        ----------
        gcs_path : str
            Path to SWC files located in a GCS bucket.

        Returns
        -------
        Dequeue[dict]
            List of dictionaries whose keys and values are the attribute
            names and values from an SWC file.
        """
        # List filenames
        bucket_name, prefix = util.parse_cloud_path(gcs_path)
        swc_paths = util.list_gcs_filenames(bucket_name, prefix, ".swc")
        zip_paths = util.list_gcs_filenames(bucket_name, prefix, ".zip")

        # Call reader
        if len(swc_paths) > 0:
            return self.read_from_gcs_swcs(bucket_name, swc_paths)
        if len(zip_paths) > 0:
            return self.read_from_gcs_zips(bucket_name, zip_paths)

        # Error
        raise Exception(f"GCS Pointer is invalid {gcs_path}")

    def read_from_gcs_swcs(self, bucket_name, swc_paths):
        """
        Reads SWC files stored in a GCS bucket.

        Parameters
        ----------
        bucket_name : str
            Name of GCS bucket containing SWC files.
        swc_paths : List[str]
            Paths to SWC files.

        Returns
        -------
        swc_dicts : Dequeue[dict]
            List of dictionaries whose keys and values are the attribute
            names and values from an SWC file.
        """
        pbar = tqdm(total=len(swc_paths), desc="Read SWCs")
        with ThreadPoolExecutor() as executor:
            # Assign threads
            threads = list()
            for path in swc_paths:
                threads.append(
                    executor.submit(self.read_from_gcs_swc, bucket_name, path)
                )

            # Store results
            swc_dicts = deque()
            for thread in as_completed(threads):
                result = thread.result()
                if result:
                    swc_dicts.append(result)
                pbar.update(1)
        return swc_dicts

    def read_from_gcs_swc(self, bucket_name, path):
        """
        Reads a single SWC file stored in a GCS bucket.

        Parameters
        ----------
        bucket_name : str
            Name of GCS bucket containing SWC files.
        swc_path : str
            Path to SWC file to be read.

        Returns
        -------
        swc_dict : dict
            Dictionaries whose keys and values are the attribute names and
            values from an SWC file.
        """
        # Initialize cloud reader
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(path)

        # Parse swc contents
        content = blob.download_as_text().splitlines()
        if len(content) > self.min_size - 10:
            swc_dict = self.parse(content)
            swc_dict["swc_name"] = get_swc_name(path)
            return swc_dict
        else:
            return False

    def read_from_gcs_zips(self, bucket_name, zip_paths):
        """
        Reads SWC files from ZIP archives stored in a GCS bucket.

        Parameters
        ----------
        bucket_name : str
            Name of GCS bucket containing SWC files.
        zip_paths : List[str]
            Paths to ZIP archives in a GCS bucket.

        Returns
        -------
        swc_dicts : Dequeue[dict]
            List of dictionaries whose keys and values are the attribute
            names and values from an SWC file.
        """
        # Initializations
        batch_size = 1000
        pbar = tqdm(total=len(zip_paths), desc="Read SWCs")

        # Main
        swc_dicts = deque()
        with ProcessPoolExecutor() as executor:
            for i in range(0, len(zip_paths), batch_size):
                # Assign processes
                processes = list()
                for zip_path in zip_paths[i:i+batch_size]:
                    processes.append(
                        executor.submit(
                            self.read_from_gcs_zip,
                            bucket_name,
                            zip_path
                        )
                    )

                # Store results
                for process in as_completed(processes):
                    try:
                        swc_dicts.extend(process.result())
                    except RefreshError:
                        pass
                    pbar.update(1)
        return swc_dicts

    def read_from_gcs_zip(self, bucket_name, zip_path, filenames=None):
        """
        Reads SWC files stored in a ZIP archive downloaded from a cloud
        bucket.

        Parameters
        ----------
        bucket_name : str
            Name of GCS bucket containing SWC files.
        zip_path : str
            Path to ZIP archive to be read.
        filenames : None or List[str], optional
            Filenames to be read if provided. Default is None.

        Returns
        -------
        swc_dicts : Dequeue[dict]
            List of dictionaries whose keys and values are the attribute
            names and values from an SWC file.
        """
        try:
            # Download zip
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            zip_content = bucket.blob(zip_path).download_as_bytes()
        except TransportError:
            print(f"Failed to read {zip_path}!")
            return deque()

        # Process files
        swc_dicts = deque()
        with ZipFile(BytesIO(zip_content), "r") as zip_file:
            filenames = zip_file.namelist() if filenames is None else filenames
            for filename in filenames:
                result = self.read_from_zipped_file(zip_file, filename)
                if result:
                    swc_dicts.append(result)
        return swc_dicts

    # --- Process content ---
    def parse(self, content):
        """
        Parses an SWC file to extract the content which is stored in a dict.
        Note that node_ids from SWC are reindex from 0 to n-1 where n is the
        number of nodes in the SWC file.

        Parameters
        ----------
        content : List[str]
            List of strings such that each is a line from an SWC file.

        Returns
        -------
        swc_dict : dict
            Dictionaries whose keys and values are the attribute names
            and values from an SWC file.
        """
        # Initializations
        content, offset = self.process_content(content)
        swc_dict = {
            "id": np.zeros((len(content)), dtype=int),
            "radius": np.zeros((len(content)), dtype=np.float16),
            "pid": np.zeros((len(content)), dtype=int),
            "xyz": np.zeros((len(content), 3), dtype=np.float32),
            "soma_nodes": set(),
        }

        # Parse content
        for i, line in enumerate(content):
            parts = line.split()
            swc_dict["id"][i] = parts[0]
            swc_dict["radius"][i] = float(parts[-2])
            swc_dict["pid"][i] = parts[-1]
            swc_dict["xyz"][i] = self.read_xyz(parts[2:5], offset)
            if int(parts[1]) == 1:
                swc_dict["soma_nodes"].add(parts[0])

        # Convert radius from nanometers to microns
        if swc_dict["radius"][0] > 100:
            swc_dict["radius"] /= 1000
        return swc_dict

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
            List of strings representing the lines of text starting from the
            line immediately after the last commented line.
        offset : List[float]
            Offset used to shift coordinates.
        """
        offset = (0, 0, 0)
        for i, line in enumerate(content):
            if line.startswith("# OFFSET"):
                offset = self.read_xyz(line.split()[2:5])
            if not line.startswith("#") and len(line) > 0:
                return content[i:], offset

    def read_xyz(self, xyz_str, offset=(0, 0, 0)):
        """
        Reads a 3D coordinate from a string and transforms it.

        Parameters
        ----------
        xyz_str : str
            Coordinate stored as a str.
        offset : List[float], optional
            Shift applied to coordinate. Default is (0, 0, 0).

        Returns
        -------
        List[float]
            Coordinate of node from an SWC file.
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
    zip_writer = ZipFile(zip_path, write_mode)
    for i, xyz in enumerate(points):
        filename = prefix + str(i + 1) + ".swc"
        to_zipped_point(zip_writer, filename, xyz, color=color, radius=radius)


def to_zipped_point(zip_writer, filename, xyz, color=None, radius=5):
    """
    Writes a point to an SWC file format, which is then stored in a ZIP
    archive.

    Parameters
    ----------
    zip_writer : zipfile.ZipFile
        ZipFile object that will store the generated SWC file.
    filename : str
        Filename of SWC file.
    xyz : ArrayLike
        Point to be written to SWC file.
    color : str, optional
        Color of nodes. Default is None.
    radius : float, optional
        Radius of point. Default is 5um.
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
        zip_writer.writestr(filename, text_buffer.getvalue())


# --- Helpers ---
def get_segment_id(swc_name):
    """
    Extract the segment ID from an SWC filename.

    Parameters
    ----------
    swc_name : str
        SWC filename, expected to be in the format "{segment_id}.swc".

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
    Gets name of the SWC file loacted at the given path, minus the extension.

    Parameters
    ----------
    path : str
        Path to SWC file.

    Returns
    -------
    name : str
        Name of the SWC file, minus the extension.
    """
    filename = os.path.basename(path)
    name, ext = os.path.splitext(filename)
    return name


def to_graph(swc_dict, set_attrs=False):
    """
    Converts an SWC dict to a NetworkX graph with reindexed nodes.

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
    graph = nx.Graph(swc_name=swc_dict["swc_name"])
    graph.add_edges_from(edges)
    if set_attrs:
        graph.graph["xyz"] = swc_dict["xyz"]
        graph.graph["radius"] = swc_dict["radius"]
    return graph
