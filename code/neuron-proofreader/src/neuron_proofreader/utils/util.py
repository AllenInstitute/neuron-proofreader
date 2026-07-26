"""
Created on Sun July 16 14:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Miscellaneous helper routines.

"""

from botocore import UNSIGNED
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor
from google.cloud import storage
from io import BytesIO
from random import sample
from tqdm import tqdm
from zipfile import ZipFile

import boto3
import json
import os
import psutil
import shutil


# --- OS Utils ---
def listdir(path, extension=None):
    """
    Lists all files in the directory at "path". If an extension is
    provided, then only files containing "extension" are returned.

    Parameters
    ----------
    path : str
        Path to directory to be searched.
    extension : str, optional
       Extension of file type of interest. The default is None.

    Returns
    -------
    filenames : List[str]
        Filenames in directory with extension "extension" if provided.
        Otherwise, list of all files in directory.
    """
    filenames = [f for f in os.listdir(path) if not f.startswith(".")]
    if extension:
        return [f for f in filenames if f.endswith(extension)]
    else:
        return filenames


def list_files_in_zip(zip_content):
    """
    Lists files in a ZIP archive.

    Parameters
    ----------
    zip_content : str
        Content stored in a ZIP archive in the form of a string of bytes.

    Returns
    -------
    List[str]
        Filenames in ZIP archive.
    """
    with ZipFile(BytesIO(zip_content), "r") as zip_file:
        return zip_file.namelist()


def list_paths(dir_path, extension=""):
    """
    Lists all paths within "directory" that end with "extension" if provided.

    Parameters
    ----------
    dir_path : str
        Path to directory to be searched.
    extension : str, optional
        If provided, only paths of files with the extension are returned.
        Default is an empty string.

    Returns
    -------
    paths : List[str]
        List of all paths within "directory".
    """
    if is_gcs_path(dir_path):
        return list_gcs_paths(dir_path, extension=extension)
    elif is_s3_path(dir_path):
        return list_s3_paths(dir_path, extension)
    else:
        filenames = listdir(dir_path, extension=extension)
        return [os.path.join(dir_path, f) for f in filenames]


def list_subdirs(path, keyword=None, return_paths=False):
    """
    List of all subdirectories at "path". If "keyword" is provided, then only
    subdirectories containing "keyword" are contained in list.

    Parameters
    ----------
    path : str
        Path to directory to be searched.
    keyword : str, optional
        Only subdirectories containing "keyword" are returned. Default is
        None.
    return_paths : bool
        Indication of whether to return full path of subdirectories.
        Default is False.

    Returns
    -------
    List[str]
        List of all subdirectories at "path".
    """
    subdirs = list()
    for subdir in os.listdir(path):
        is_dir = os.path.isdir(os.path.join(path, subdir))
        is_hidden = subdir.startswith(".")
        if is_dir and not is_hidden:
            subdir = os.path.join(path, subdir) if return_paths else subdir
            if (keyword and keyword in subdir) or not keyword:
                subdirs.append(subdir)
    return sorted(subdirs)


def mkdir(dir_path, delete=False):
    """
    Creates a directory at "path".

    Parameters
    ----------
    path : str
        Path of directory to be created.
    delete : bool, optional
        Indication of whether to delete directory at path if it already
        exists. Default is False.
    """
    if delete:
        rmdir(dir_path)
    os.makedirs(dir_path, exist_ok=True)


def rmdir(dir_path):
    """
    Removes directory and all subdirectories at "path".

    Parameters
    ----------
    dir_path : str
        Path to directory and subdirectories to be deleted if they exist.
    """
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)


def set_filename_in_zip(zipfile, name):
    """
    Sets the path for a file within a ZIP archive. If a file with the same
    name already exists, then this routine finds a suffix to append to the
    filename.

    Parameters
    ----------
    zipfile : ZipFile.zipfile
        Zip file object used to write SWC files to a ZIP archive.
    name : str
        Name of SWC file to be write SWC files to a ZIP archive.

    Returns
    -------
    filename : str
        Name of SWC file to be written to ZIP archive.
    name : str
        Name of SWC file to be  archive.
    """
    cnt = 0
    filename = f"{name}.swc"
    while filename in zipfile.namelist():
        filename = f"{name}.{cnt}.swc"
        cnt += 1
    return filename


# --- IO Utils ---
def combine_zips(zip_paths, output_zip_path):
    """
    Combines a list of ZIP archives into a single ZIP archive.

    Parameters
    ----------
    zip_paths : List[str]
        List of ZIP archieves to be combined.
    output_zip_path : str
        Path to ZIP archive to be written.
    """
    seen_files = set()
    with ZipFile(output_zip_path, "w") as out_zip:
        for zip_path in tqdm(zip_paths, desc="Combine ZIPs"):
            with ZipFile(zip_path, "r") as zip_in:
                for item in zip_in.infolist():
                    if item.filename not in seen_files:
                        seen_files.add(item.filename)
                        out_zip.writestr(item, zip_in.read(item.filename))
                    else:
                        print("File Conflict:", item.filename)


def read_json(path):
    """
    Reads JSON file located at the given path.

    Parameters
    ----------
    path : str
        Path to JSON file to be read.

    Returns
    -------
    dict
        Contents of JSON file.
    """
    if is_gcs_path(path):
        return json.loads(read_gcs_txt(path))
    else:
        with open(path, "r") as f:
            return json.load(f)


def read_txt(path, client=None):
    """
    Reads txt file at the given path.

    Parameters
    ----------
    path : str
        Path to txt file.

    Returns
    -------
    str
        Text from the txt file.
    """
    if is_s3_path(path):
        return read_s3_txt(path, client=client)
    elif is_gcs_path(path):
        return read_gcs_txt(path, client=client)
    else:
        with open(path, "r") as f:
            return f.read()


def read_zip(zip_file, path):
    """
    Reads txt file located in a ZIP archive.

    Parameters
    ----------
    zip_file : ZipFile
        ZIP archive containing txt file to be read.
    path : str
        Path to txt file within ZIP archive to be read.

    Returns
    -------
    str
        Contents of text file in ZIP archive.
    """
    with zip_file.open(path) as f:
        return f.read().decode("utf-8")


def update_txt(path, text, verbose=True):
    """
    Appends the given text to a specified text file and prints the text.

    Parameters
    ----------
    path : str
        Path to txt file where the text will be appended.
    text : str
        Text to be written to the file.
    verbose : bool, optional
        Indication of whether to printout text. Default is True.
    """
    # Printout text (if applicable)
    if verbose:
        print(text)

    # Update txt file
    with open(path, "a") as file:
        file.write(text + "\n")


def write_json(path, contents):
    """
    Writes "contents" to a JSON file at "path".

    Parameters
    ----------
    path : str
        Path that txt file is written to.
    contents : dict
        Contents to be written to a JSON file.
    """
    with open(path, "w") as f:
        json.dump(contents, f)


def write_list(path, my_list):
    """
    Writes each item in a list to a text file, with each item on a new line.

    Parameters
    ----------
    path : str
        Path where text file is to be written.
    my_list : list
        The list of items to write to the file.
    """
    with open(path, "w") as file:
        for item in my_list:
            file.write(f"{item}\n")


def write_txt(path, contents):
    """
    Writes "contents" to a txt file at "path".

    Parameters
    ----------
    path : str
        Path that txt file is written to.
    contents : str
        String to be written to txt file.
    """
    f = open(path, "w")
    f.write(contents)
    f.close()


# --- Cloud Utils ---
def get_google_swcs_prefix(root_prefix, brain_id, segmentation_id):
    # Determine old vs. new result
    prefix1 = os.path.join(root_prefix, brain_id, "whole_brain")
    prefix2 = os.path.join(root_prefix, "whole_brain", brain_id)
    if check_gcs_prefix_exists(prefix1):
        prefix = prefix1
    elif check_gcs_prefix_exists(prefix2):
        prefix = prefix2
    else:
        raise Exception("Unable to find Google swcs result!")

    # Get SWC dirname
    prefix = os.path.join(prefix, segmentation_id)
    dirname = get_google_swcs_dirname(prefix)
    return os.path.join(prefix, dirname)


def get_google_swcs_dirname(prefix):
    for subprefix in list_gcs_subprefixes(prefix):
        dirname = subprefix.split("/")[-2]
        if "swc" in dirname:
            return dirname
    return "swcs"


def parse_cloud_path(path):
    """
    Parses a cloud storage path into its bucket name and key/prefix. Supports
    paths of the form: "{scheme}://bucket_name/prefix" or without a scheme.

    Parameters
    ----------
    path : str
        Path to be parsed.

    Returns
    -------
    bucket_name : str
        Name of the bucket.
    prefix : str
        Cloud prefix.
    """
    # Remove s3:// or gs:// if present
    if path.startswith("s3://") or path.startswith("gs://"):
        path = path[len("s3://"):]

    # Split path
    parts = path.split("/", 1)
    bucket_name = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return bucket_name, key


# --- GCS Utils ---
def check_gcs_file_exists(path):
    """
    Checks if a file exists at the given GCS path.

    Parameters
    ----------
    path : str
        GCS path to check.

    Returns
    -------
    bool
        Indication of whether the path exists.
    """
    bucket_name, key = parse_cloud_path(path)
    bucket = storage.Client().bucket(bucket_name)
    return bucket.blob(key).exists()


def check_gcs_prefix_exists(prefix):
    bucket_name, key = parse_cloud_path(prefix)
    bucket = storage.Client().bucket(bucket_name)
    key = key.rstrip("/") + "/"
    exists = any(bucket.list_blobs(prefix=key, max_results=1))
    return exists


def is_gcs_path(path):
    """
    Checks if the path is a GCS path.

    Parameters
    ----------
    path : str
        Path to be checked.

    Returns
    -------
    bool
        Indication of whether the path is a GCS path.
    """
    return path.startswith("gs://")


def list_gcs_paths(path, extension=""):
    """
    Lists paths at a GCS prefix with the given extension.

    Parameters
    ----------
    path : str
        Path to location in a GCS bucket.
    extension : str, optional
        File extension of filenames to be listed. Default is an empty string.

    Returns
    -------
    List[str]
        Paths under the GCS prefix with the given extension.
    """
    # Create bucket
    bucket_name, prefix = parse_cloud_path(path)
    bucket = storage.Client().bucket(bucket_name)

    # List paths
    paths = list()
    for name in [b.name for b in bucket.list_blobs(prefix=prefix)]:
        if extension in name:
            paths.append(os.path.join(f"gs://{bucket_name}", name))
    return sorted(paths)


def list_gcs_subprefixes(path):
    """
    Lists all direct subdirectories of a given location in a GCS bucket.

    Parameters
    ----------
    path : str
        Path to location in a GCS bucket.

    Returns
    -------
    List[str]
         Direct subdirectories.
    """
    bucket_name, prefix = parse_cloud_path(path)
    prefix = prefix.rstrip("/") + "/"

    # Load blobs
    blobs = storage.Client().list_blobs(
        bucket_name, prefix=prefix, delimiter="/"
    )
    [blob.name for blob in blobs]

    # Parse directory contents
    prefix_depth = len(prefix.split("/"))
    subdirs = list()
    for prefix in blobs.prefixes:
        is_dir = prefix.endswith("/")
        is_direct_subdir = len(prefix.split("/")) - 1 == prefix_depth
        if is_dir and is_direct_subdir:
            subdirs.append(prefix)
    return subdirs


def read_gcs_txt(prefix, client=None):
    """
    Reads a txt file stored in a GCS bucket.

    Parameters
    ----------
    path : str
        Path to txt file to be read.

    Returns
    -------
    str
        Contents of txt file.
    """
    bucket_name, subprefix = parse_cloud_path(prefix)
    client = client or storage.Client()
    bucket = client.bucket(bucket_name)
    return bucket.blob(subprefix).download_as_text()


# --- S3 Utils ---
def is_s3_path(path):
    """
    Checks if the path is an S3 path.

    Parameters
    ----------
    path : str
        Path to be checked.

    Returns
    -------
    bool
        Indication of whether the path is an S3 path.
    """
    return path.startswith("s3://")


def list_s3_paths(path, extension=""):
    """
    Lists all object keys in a public S3 bucket under a given prefix,
    optionally filters by file extension.

    Parameters
    ----------
    path : str
        Path to location in an S3 bucket.
    extension : str, optional
        File extension to filter by. Default is an empty string.

    Returns
    -------
    paths : List[str]
        S3 object keys that match the prefix and extension filter.
    """
    # Create an anonymous client for public buckets
    bucket_name, prefix = parse_cloud_path(path)
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    # List all objects under the prefix
    paths = list()
    if "Contents" in response:
        for obj in response["Contents"]:
            filename = obj["Key"]
            if filename.endswith(extension):
                path = os.path.join(f"s3://{bucket_name}", filename)
                paths.append(path)
    return paths


def read_s3_txt(prefix, client=None):
    """
    Reads a txt file stored in an S3 bucket.

    Parameters
    ----------
    prefix : str
        Path to txt file to be read.

    Returns
    -------
    str
        Contents of txt file.
    """
    bucket_name, subprefix = parse_cloud_path(prefix)
    client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    obj = client.get_object(Bucket=bucket_name, Key=subprefix)
    return obj["Body"].read().decode("utf-8")


def upload_dir_to_s3(dir_path, bucket_name, prefix):
    """
    Uploads a directory on the local machine to an S3 bucket.

    Parameters
    ----------
    dir_path : str
        Path to directory to be written to S3.
    bucket_name : str
        Name of S3 bucket.
    prefix : str
        Path within S3 bucket.
    """
    # Upload files
    with ThreadPoolExecutor() as executor:
        for name in os.listdir(dir_path):
            src_path = os.path.join(dir_path, name)
            dst_path = os.path.join(prefix, name)
            if os.path.isdir(src_path):
                subprefix = os.path.join(prefix, name)
                upload_dir_to_s3(src_path, bucket_name, subprefix)
            else:
                executor.submit(
                    upload_file_to_s3(src_path, bucket_name, dst_path)
                )


def upload_file_to_s3(src_path, bucket_name, dst_path):
    """
    Writes a single file on the local machine to an S3 bucket.

    Parameters
    ----------
    src_path : str
        Path to file to be written to S3.
    bucket_name : str
        Name of S3 bucket.
    dst_path : str
        Path within S3 bucket that source file is to be written to.
    """
    s3 = boto3.client("s3")
    s3.upload_file(src_path, bucket_name, dst_path)


# --- Dictionary Utils ---
def find_best(my_dict, maximize=True):
    """
    Finds the key associated with the largest integer or longest list.

    Parameters
    ----------
    my_dict : dict
        Dictionary to be searched.
    maximize : bool, optional
        Indication of whether to find the largest/longest or
        smallest/shortest.

    Returns
    -------
    hashable
        Key associated with the longest list or largest integer in "my_dict".
    """

    def score(v):
        """
        Assigns a score to a given value.
        """
        return v if isinstance(v, (int, float)) else len(v)

    optimize = max if maximize else min
    return optimize(my_dict, key=lambda k: score(my_dict[k]))


def remove_items(my_dict, keys):
    """
    Removes dictionary items corresponding to "keys".

    Parameters
    ----------
    my_dict : dict
        Dictionary to be edited.
    keys : list
        List of keys to be deleted from "my_dict".

    Returns
    -------
    dict
        Updated dictionary.
    """
    return {k: v for k, v in my_dict.items() if k not in keys}


# --- Miscellaneous ---
def get_memory_usage():
    """
    Gets the current memory usage in gigabytes.

    Returns
    -------
    float
        Current memory usage in gigabytes.
    """
    return psutil.virtual_memory().used / 1e9


def numpy_to_hashable(arr):
    """
    Converts a numpy array to a hashable data structure.

    Parameters
    ----------
    arr : numpy.ndarray
        Array to be converted.

    Returns
    -------
    list
        Hashable items from "arr".
    """
    return [tuple(item) for item in arr.tolist()]


def sample_once(my_container):
    """
    Samples a single element from the given container.

    Parameters
    ----------
    my_container : container
        Container to be sampled from.

    Returns
    -------
    hashable
        Single element from the given container.
    """
    if not isinstance(my_container, list):
        return sample(list(my_container), 1)[0]
    else:
        return sample(my_container, 1)[0]


def time_writer(t, unit="seconds"):
    """
    Converts a runtime to a larger unit of time if applicable.

    Parameters
    ----------
    t : float
        Runtime.
    unit : str, optional
        Unit that the given time is expressed in. Default is "seconds".

    Returns
    -------
    float
        Runtime
    str
        Unit of time.
    """
    assert unit in ["seconds", "minutes", "hours"]
    upd_unit = {"seconds": "minutes", "minutes": "hours"}
    if t < 60 or unit == "hours":
        return t, unit
    else:
        t /= 60
        unit = upd_unit[unit]
        t, unit = time_writer(t, unit=unit)
    return t, unit
