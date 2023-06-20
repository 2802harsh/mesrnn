#!/usr/bin/env python

"""
What does this file do?
"""

__author__ = "Aamir Hasan"
__version__ = "1.0"
__email__ = "hasanaamir215@gmail.com"

from os import listdir
from os.path import join, exists, isdir
from torch import cuda, from_numpy
from torch.utils.data import Dataset
from pickle import load


class TrajectoryDataset(Dataset):
    # __datasets__ = ["eth_univ", "eth_hotel", "ucy_zara01", "ucy_zara02", "ucy_univ"]
    __datasets__ = ["eth_univ", "eth_hotel"]
    # __datasets__ = ["biwi_hotel", "crowds_students003", "crowds_students001", "crowds_zara01", "crowds_zara03", "lcas", "wildtrack"]

    def __init__(self, test_dataset, root_dir, test=False):
        """
        :param test_dataset: name of the test dataset
        :type test_dataset: int
        :param root_dir: root directory containing all the data
        :type root_dir: string
        """
        # make sure that the dataset directories exist
        for dataset in self.__datasets__:
            if not exists(join(root_dir, dataset, "pre_processed")):
                raise ValueError("Data has not been preprocessed! Please run preprocessor.py")

        # check to make sure the test_dataset index is valid
        if test_dataset < 0 or test_dataset > 2:
            raise ValueError("Invalid test dataset. Must be between 0 and 4.")

        # load all the paths for the scenes
        self.scene_paths = []
        for i in range(len(self.__datasets__)):
            dataset_path = join(root_dir, self.__datasets__[i], "pre_processed")
            if test:
                if i == test_dataset:
                    self.scene_paths += [join(dataset_path, f) for f in listdir(dataset_path) if f[-4:] == ".pkl"]
                    break
            else:
                self.scene_paths += [join(dataset_path, f) for f in listdir(dataset_path) if f[-4:] == ".pkl"]
        print("Loaded datasets.")

    def __len__(self):
        return len(self.scene_paths)

    def __getitem__(self, idx):
        device = 'cuda' if cuda.is_available() else 'cpu'

        # load actual trajectories from the scene and return it
        trajectories, last_valid_idx, num_peds  = load(open(self.scene_paths[idx], 'rb'))

        return from_numpy(trajectories).to(device), last_valid_idx, num_peds, [], [], -1, \
               1, idx, 9


class TrajNetDataset(Dataset):
    def __init__(self, root_dir, remove_ds):
        """
        :param root_dir: root directory containing all the data
        :type root_dir: string
        """
        # make sure that the dataset directories exist
        if not exists(join(root_dir)):
            raise ValueError("Invalid dataset path")

        ds_to_remove = remove_ds.split(" ")

        # load all the paths for the scenes
        self.scene_paths = []
        self.scene_path_to_id = {}
        
        for dir in listdir(root_dir):
            if isdir(join(root_dir, dir)) and dir[:-7] not in ds_to_remove:
                # took out list comprehension to create dictionary
                for f in listdir(join(root_dir, dir)):
                    if f[-4:] == ".pkl":
                        scene_path = join(root_dir, dir, f)
                        self.scene_paths += [scene_path]
                        self.scene_path_to_id[scene_path] = int(f[:-4])
               
            
                # self.scene_paths += [join(root_dir, dir, f) for f in listdir(join(root_dir, dir)) if f[-4:] == ".pkl"]
                # self.scene_path_to_id[]

        print("Loaded datasets.")

    def __len__(self):
        return len(self.scene_paths)

    def __getitem__(self, idx):
        device = 'cuda' if cuda.is_available() else 'cpu'

        # load actual trajectories from the scene and return it
        trajectories, last_valid_idx, num_peds, _, ped_ids, min_val, max_val, scene_id, dataset_name = load(open(self.scene_paths[idx], 'rb'))
        scene_id = self.scene_path_to_id[self.scene_paths[idx]]
        
        # with open(self.scene_paths[idx], 'rb') as f:
        #     data = load(f)
        #     print("REAL DATA for scene ", scene_id)
        #     print("----------------------------------------------------")
        #     print(data)
            
        return trajectories.to(device), last_valid_idx, num_peds, 0, ped_ids, min_val, max_val, scene_id, dataset_name
