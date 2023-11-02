from torch.utils.data import Dataset
import torch
import numpy as np

class ConcatDatasets(Dataset):
    def __init__(self,dataset_unpaired,dataset_paired):
        '''
        Concat the unpaired dataset and paired dataset to a single dataset
        '''
        self.dataset_unpaired = dataset_unpaired
        self.dataset_paired = dataset_paired
        self.len_unpaired = len(self.dataset_unpaired)
        self.len_paired = len(self.dataset_paired)

    def __len__(self):
        return self.len_unpaired + self.len_paired

    def __getitem__(self, item):
        if item < self.len_unpaired:
            return self.dataset_unpaired[item]
        else:
            return self.dataset_paired[item-self.len_unpaired]


class PairedDataset(Dataset):
    def __init__(self,source_paired_dataset,target_paired_dataset):
        '''
        The images in the two datasets are correspondingly paired.
        :param source_paired_dataset: paired source dataset, return (image,label)
        :param target_paired_dataset: paired target dataset, return (image,label)
        '''
        self.source_paired_dataset = source_paired_dataset
        self.target_paired_dataset = target_paired_dataset

    def __len__(self):
        return len(self.source_paired_dataset)

    def __getitem__(self, item):
        '''
        :return: paired source and target images
        '''
        return self.source_paired_dataset[item][0],self.target_paired_dataset[item][0]

class UnPairedDataset(Dataset):
    def __init__(self,source_dataset,target_dataset,non_zero_dict_path="exp/OT/models/non_zero_dict_1e-05.pkl"):
        '''
        :param source_dataset: source dataset (containing paired and unpaired)
        :param target_dataset: target dataset (containing paired and unpaired)
        :param non_zero_dict_path: path to the stored dict of non-zero H, elements are like {i:(indexes,values of H)}
        '''
        self.non_zero_dict = torch.load(non_zero_dict_path)
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset

    def __getitem__(self, source_index):
        non_zero_indexes,values = self.non_zero_dict[source_index]
        if len(non_zero_indexes) == 0:
            returns = self.__getitem__((source_index+1)%self.__len__())
            return returns
        else:
            values = values/values.sum()
            target_index = np.random.choice(non_zero_indexes,p=values)
            return self.source_dataset[source_index][0],self.target_dataset[target_index][0]

    def __len__(self):
        return len(self.non_zero_dict)


