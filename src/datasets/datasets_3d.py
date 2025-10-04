from typing import Optional, Callable, List, Dict, Tuple

import os
from tqdm import tqdm
import glob
import ase
import numpy as np

import torch
from torch_geometric.data import (InMemoryDataset, download_url, extract_zip, Data)
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.data.data import Data
from torch_geometric.data.batch import Batch
from torch_geometric.data.data import DataEdgeAttr


class PCQM4MV2_XYZ(InMemoryDataset):
    '''3D coordinates for molecules in the PCQM4Mv2 dataset (from zip).'''

    raw_url = 'http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2_xyz.zip'

    def __init__(
        self, root: str, transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None, dataset_arg: Optional[str] = None
    ):
        '''
        Args:
            root (str): Root directory where the dataset should be saved.
            transform (Callable, optional): A function/transform that takes in an
                :obj:`torch_geometric.data.Data` object and returns a transformed
                version. The data object will be transformed before every access.
                (default: :obj:`None`)
            pre_transform (Callable, optional): A function/transform that takes in
                an :obj:`torch_geometric.data.Data` object and returns a
                transformed version. The data object will be transformed before
                being saved to disk. (default: :obj:`None`)
            pre_filter (Callable, optional): A function that takes in an
                :obj:`torch_geometric.data.Data` object and returns a boolean
                value, indicating whether the data object should be included in the
                final dataset. (default: :obj:`None`)
            dataset_arg (str, optional): Not used for this dataset. Included for compatibility.
                (default: :obj:`None`) 
        '''
        assert dataset_arg is None, "PCQM4MV2 does not take any dataset args."
        super().__init__(root, transform, pre_transform, pre_filter)
        with torch.serialization.safe_globals([GlobalStorage, Data, Batch, DataEdgeAttr]):
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
            # self.data : Data(pos=[99627033, 3], z=[99627033], idx=[3378606])
            # self.slices : defaultdict( <class 'dict'>, 
            # {'pos': tensor([       0,       30,       64,  ..., 99626964, 99626998, 99627033]), 
            #  'z': tensor([       0,       30,       64,  ..., 99626964, 99626998, 99627033]), 
            #  'idx': tensor([      0,       1,       2,  ..., 3378604, 3378605, 3378606])} )

    @property
    def raw_file_names(self) -> List[str]:
        '''Returns the names of the raw files that should be downloaded.'''
        return ['pcqm4m-v2_xyz']

    @property
    def processed_file_names(self) -> str:
        '''Returns the name of the processed file that will be created.'''
        return 'pcqm4mv2__xyz.pt'

    def download(self) -> None:
        '''Downloads the raw data from the URL and extracts it to the raw directory.'''
        file_path = download_url(self.raw_url, self.raw_dir)
        extract_zip(file_path, self.raw_dir)
        os.unlink(file_path)

    def process(self) -> None:
        '''Processes the raw data and saves it to the processed directory.'''
        dataset = PCQM4MV2_3D(self.raw_paths[0])
        
        data_list = []
        for i, mol in enumerate(tqdm(dataset)):
            pos = mol['coords']
            pos = torch.tensor(pos, dtype=torch.float)
            z = torch.tensor(mol['atom_type'], dtype=torch.long)

            data = Data(z=z, pos=pos, idx=i)

            # test print
            # if i % 100000 == 1:
            #     print(f'Processing molecule {i}, pos = {pos}, z = {z}, idx = {i}')
            #     print(f'data = {data}')
            #     print(f'data.pos = {data.pos}, data.z = {data.z}, data.idx = {data.idx}')

            # Example output:
            # z = tensor([ 6,  6,  6,  6,  7,  7,  8, 16,  1,  1,  1,  1,  1,  1])
            # pos = tensor([[ 4.5408,  0.0705,  0.8834],
            #               [ 2.8599, -2.2980, -1.6083], ..., [ 4.6509, -1.2351, -4.0150]])
            # idx = 400001
            # data = Data(pos=[14, 3], z=[14], idx=400001)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0], pickle_protocol=4)
        



class PCQM4MV2_3D:
    '''Data loader for PCQM4MV2 from raw xyz files. Loads data given a path with .xyz files.'''
    
    def __init__(self, path: str):
        '''
        Args:
            path (str): Path to the directory containing .xyz files.
        '''
        self.path = path
        self.xyz_files = glob.glob(path + '/*/*.xyz')
        self.xyz_files = sorted(self.xyz_files, key=self._molecule_id_from_file)
        self.num_molecules = len(self.xyz_files)
        
    def read_xyz_file(self, file_path: str) -> Dict[str, np.ndarray]:
        '''
        Reads an XYZ file and returns a dictionary with atom types and coordinates.
        Args:
            file_path (str): Path to the XYZ file.
        Returns:
            Dict[str, np.ndarray]: A dictionary with keys 'atom_type' and 'coords'.
        '''
        atom_types = np.genfromtxt(file_path, skip_header=1, usecols=range(1), dtype=str)
        atom_types = np.array([ase.Atom(sym).number for sym in atom_types])
        atom_positions = np.genfromtxt(file_path, skip_header=1, usecols=range(1, 4), dtype=np.float32)        
        return {'atom_type': atom_types, 'coords': atom_positions}
    
    def _molecule_id_from_file(self, file_path: str) -> int:
        '''
        Extracts the molecule ID from the file name.
        Args:
            file_path (str): Path to the XYZ file.
        Returns:
            int: The molecule ID extracted from the file name.
        '''
        return int(os.path.splitext(os.path.basename(file_path))[0])
    
    def __len__(self) -> int:
        '''
        Returns the number of molecules in the dataset.
        Returns:
            int: The number of molecules in the dataset.
        '''
        return self.num_molecules
    
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        '''
        Gets the data for a specific molecule by its index.
        Args:
            idx (int): Index of the molecule.
        Returns:
            Dict[str, np.ndarray]: A dictionary with keys 'atom_type' and 'coords' for the molecule.
        '''
        return self.read_xyz_file(self.xyz_files[idx])