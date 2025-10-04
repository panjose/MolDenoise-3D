import json
import warnings
from rdkit import RDLogger
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch


RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")


def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    '''
    Reads a JSON Lines file and returns a list of dictionaries.
    Args:
        file_path (str): Path to the JSON Lines file.
    Returns:
        List[Dict[str, Any]]: A list of dictionaries where each dictionary represents a line in the file.
    '''
    data_dict = []
    with open(file_path, "r", encoding="utf8") as file:
        for line in file:
            data = json.loads(line.strip())
            data_dict.append(data)
    return data_dict


def write_jsonl(file_path: str, data_dict: List[Dict[str, Any]]) -> None:
    '''
    Writes a list of dictionaries to a JSON Lines file.
    Args:
        file_path (str): Path to the JSON Lines file.
        data_dict (List[Dict[str, Any]]): A list of dictionaries to write to the file.
    '''
    with open(file_path, "w+", encoding="utf8") as file:
        for data in data_dict:
            file.write(json.dumps(data) + "\n")


def read_json(file_path: str) -> Dict[str, Any] | List[Dict[str, Any]]:
    ''' 
    Reads a JSON file and returns its content.
    Args:
        file_path (str): Path to the JSON file.
    Returns:
        Dict[str, Any] | List[Dict[str, Any]]: The content of the JSON file.
    '''
    with open(file_path, "r", encoding="utf8") as file:
        return json.load(file)


def write_json(file_path: str, data_dict: Dict[str, Any]) -> None:
    '''
    Writes a dictionary to a JSON file.
    Args:
        file_path (str): Path to the JSON file.
        data_dict (Dict[str, Any]): The dictionary to write to the file.
    '''
    with open(file_path, "w+", encoding="utf8") as file:
        file.write(json.dumps(data_dict))


def train_val_test_split(
    dset_len: int, train_size: Optional[float], val_size: Optional[float], test_size: Optional[float], 
    seed: int, order: Optional[list] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Splits a dataset into training, validation, and testing sets.
    Args:
        dset_len (int): Length of the dataset.
        train_size (float or None): Size of the training set. Can be a float or an integer.
        val_size (float or None): Size of the validation set. Can be a float or an integer.
        test_size (float or None): Size of the testing set. Can be a float or an integer.
        seed (int): Random seed for reproducibility.
        order (list or None): Optional list to specify the order of indices.
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple of numpy arrays containing the indices of the 
            training, validation, and testing sets.
    '''
    assert (train_size == None) + (val_size == None) + (
        test_size == None
    ) <= 1, "Only one of train_size, val_size, test_size is allowed to be None."
    is_float = (
        isinstance(train_size, float),
        isinstance(val_size, float),
        isinstance(test_size, float),
    )

    train_size = round(dset_len * train_size) if is_float[0] else train_size
    val_size = round(dset_len * val_size) if is_float[1] else val_size
    test_size = round(dset_len * test_size) if is_float[2] else test_size

    if train_size is None:
        train_size = dset_len - val_size - test_size
    elif val_size is None:
        val_size = dset_len - train_size - test_size
    elif test_size is None:
        test_size = dset_len - train_size - val_size

    if train_size + val_size + test_size > dset_len:
        if is_float[2]:
            test_size -= 1
        elif is_float[1]:
            val_size -= 1
        elif is_float[0]:
            train_size -= 1

    assert train_size >= 0 and val_size >= 0 and test_size >= 0, (
        f"One of training ({train_size}), validation ({val_size}) or "
        f"testing ({test_size}) splits ended up with a negative size."
    )

    total = train_size + val_size + test_size
    assert dset_len >= total, (
        f"The dataset ({dset_len}) is smaller than the "
        f"combined split sizes ({total})."
    )
    if total < dset_len:
        warnings.warn(f"{dset_len - total} samples were excluded from the dataset")

    idxs = np.arange(dset_len, dtype=int)
    if order is None:
        idxs = np.random.default_rng(seed).permutation(idxs)

    idx_train = idxs[:train_size]
    idx_val = idxs[train_size : train_size + val_size]
    idx_test = idxs[train_size + val_size : total]

    if order is not None:
        idx_train = [order[i] for i in idx_train]
        idx_val = [order[i] for i in idx_val]
        idx_test = [order[i] for i in idx_test]

    return np.array(idx_train), np.array(idx_val), np.array(idx_test)


def make_splits(
    dataset_len: int, train_size: Optional[float], val_size: Optional[float], test_size: Optional[float],
    seed: int, filename: Optional[str] = None, splits : Optional[str] = None, order: Optional[list] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    Create train, validation, and test splits for a dataset.
    Args:
        dataset_len (int): The total number of samples in the dataset.
        train_size (float or None): The proportion or absolute number of samples for the training set
            (if None, the training set will be determined by the remaining samples after validation and test sets).
        val_size (float or None): The proportion or absolute number of samples for the validation set
            (if None, the validation set will be determined by the remaining samples after training and test sets).
        test_size (float or None): The proportion or absolute number of samples for the test set
            (if None, the test set will be determined by the remaining samples after training and validation sets).
        seed (int): Random seed for reproducibility.
        filename (str or None): If provided, the splits will be saved to this file in `.npz` format.
        splits (str or None): If provided, the splits will be loaded from this file in `.npz` format.
        order (list or None): If provided, the indices will be ordered according to this list.
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Three tensors containing the indices for the
            training, validation, and test sets, respectively.
    '''
    if splits is not None:
        splits = np.load(splits)
        idx_train = splits["idx_train"]
        idx_val = splits["idx_val"]
        idx_test = splits["idx_test"]
    else:
        idx_train, idx_val, idx_test = train_val_test_split(
            dataset_len, train_size, val_size, test_size, seed, order
        )

    if filename is not None:
        np.savez(filename, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)

    return (
        torch.from_numpy(idx_train),
        torch.from_numpy(idx_val),
        torch.from_numpy(idx_test),
    )

