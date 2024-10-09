# © 2022. Triad National Security, LLC. All rights reserved.

# This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos

# National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.

# Department of Energy/National Nuclear Security Administration. All rights in the program are

# reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear

# Security Administration. The Government is granted for itself and others acting on its behalf a

# nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare

# derivative works, distribute copies to the public, perform publicly and display publicly, and to permit

# others to do so.

import os
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import transforms as T

class FWIDataset(Dataset):
    ''' FWI dataset
    For convenience, in this class, a batch refers to a npy file 
    instead of the batch used during training.

    Args:
        anno: path to annotation file
        preload: whether to load the whole dataset into memory
        sample_ratio: downsample ratio for seismic data
        file_size: # of samples in each npy file
        transform_data|label: transformation applied to data or label
    '''
    def __init__(self, anno, preload=False, sample_ratio=1, file_size=500,
                    transform_data=None, transform_label=None):
        if not os.path.exists(anno):
            print(f'Annotation file {anno} does not exists')
        self.preload = preload
        self.sample_ratio = sample_ratio
        self.file_size = file_size
        self.transform_data = transform_data
        self.transform_label = transform_label
        with open(anno, 'r') as f:
            self.batches = f.readlines()
        if preload: 
            self.data_list, self.label_list = [], []
            for batch in self.batches:
                if batch == '\n':
                    break
                data, label = self.load_every(batch)
                self.data_list.append(data)
                if label is not None:
                    self.label_list.append(label)

    # Load from one line
    def load_every(self, batch):
        batch = batch.split('\t')
        data_path = batch[0] if len(batch) > 1 else batch[0][:-1]
        data = np.load(data_path)[:, :, ::self.sample_ratio, :]
        data = data.astype('float32')
        if len(batch) > 1:
            label_path = batch[1][:-1]    
            label = np.load(label_path)
            label = label.astype('float32')
        else:
            label = None
        
        return data, label
        
    def __getitem__(self, idx):
        batch_idx, sample_idx = idx // self.file_size, idx % self.file_size
        if self.preload:
            data = self.data_list[batch_idx][sample_idx]
            label = self.label_list[batch_idx][sample_idx] if len(self.label_list) != 0 else None
        else:
            data, label = self.load_every(self.batches[batch_idx])
            data = data[sample_idx]
            label = label[sample_idx] if label is not None else None
        if self.transform_data:
            data = self.transform_data(data)
        if self.transform_label and label is not None:
            label = self.transform_label(label)
        return data, label if label is not None else np.array([])
        
    def __len__(self):
        return len(self.batches) * self.file_size

# 带source的OpenFWI
class LDataset(Dataset):
    def __init__(self, anno, preload=False, sample_ratio=1, file_size=500,
                 transform_data=None, transform_source=None, transform_label=None, du=''):
        if not os.path.exists(anno):
            print(f'Annotation file {anno} does not exists')
        self.preload = preload
        self.sample_ratio = sample_ratio
        self.file_size = file_size
        self.transform_data = transform_data
        self.transform_source = transform_source
        self.transform_label = transform_label
        self.du = du
        with open(anno, 'r') as f:
            self.batches = f.readlines()
        if preload:
            self.data_list, self.source_list, self.label_list = [], [], []
            for batch in self.batches:
                if batch == '\n':
                    break
                data, source, label = self.load_every(batch)
                self.data_list.append(data)
                self.source_list.append(source)
                self.label_list.append(label)

    # Load from one line
    def load_every(self, batch):
        batch = batch.split('\t')
        data_path = batch[0]
        data = np.load(os.path.join(self.du, data_path))[:, :, ::self.sample_ratio, :]
        data = data.astype('float32')

        source_path = batch[1]
        source = np.load(os.path.join(self.du, source_path))
        source = source.astype('float32')

        label_path = batch[2][:-1]
        label = np.load(os.path.join(self.du, label_path))
        label = label.astype('float32')

        return data, source, label

    def __getitem__(self, idx):
        batch_idx, sample_idx = idx // self.file_size, idx % self.file_size
        if self.preload:
            data = self.data_list[batch_idx][sample_idx]
            source = self.source_list[batch_idx][sample_idx]
            label = self.label_list[batch_idx][sample_idx]
        else:
            data, source, label = self.load_every(self.batches[batch_idx])
            data = data[sample_idx]
            source = source[sample_idx]
            label = label[sample_idx]
        if self.transform_data:
            data = self.transform_data(data)
        if self.transform_source:
            source = self.transform_source(source)
        if self.transform_label and label is not None:
            label = self.transform_label(label)
        return data, source, label if label is not None else np.array([])

    def __len__(self):
        return len(self.batches) * self.file_size



class FLDataset(Dataset):
    def __init__(self, anno, preload=False, sample_ratio=1, file_size=500,
                 transform_data=None, transform_frequency=None, transform_source=None, transform_label=None, du=''):
        if not os.path.exists(anno):
            print(f'Annotation file {anno} does not exists')
        self.preload = preload
        self.sample_ratio = sample_ratio
        self.file_size = file_size
        self.transform_data = transform_data
        self.transform_frequency = transform_frequency
        self.transform_source = transform_source
        self.transform_label = transform_label
        self.du = du
        with open(anno, 'r') as f:
            self.batches = f.readlines()
        if preload:
            self.data_list, self.frequency_list, self.source_list, self.label_list = [], [], [], []
            for batch in self.batches:
                if batch == '\n':
                    break
                data, frequency, source, label = self.load_every(batch)
                self.data_list.append(data)
                self.frequency_list.append(frequency)
                self.source_list.append(source)
                self.label_list.append(label)

    # Load from one line
    def load_every(self, batch):
        batch = batch.split('\t')
        data_path = batch[0]
        data = np.load(os.path.join(self.du, data_path))[:, :, ::self.sample_ratio, :]
        data = data.astype('float32')

        frequency_path = batch[1]
        frequency = np.load(os.path.join(self.du, frequency_path))
        frequency.astype('float32')

        source_path = batch[2]
        source = np.load(os.path.join(self.du, source_path))
        source = source.astype('float32')

        label_path = batch[3][:-1]
        label = np.load(os.path.join(self.du, label_path))
        label = label.astype('float32')

        return data, frequency, source, label

    def __getitem__(self, idx):
        batch_idx, sample_idx = idx // self.file_size, idx % self.file_size
        if self.preload:
            data = self.data_list[batch_idx][sample_idx]
            frequency = self.frequency_list[batch_idx][sample_idx]
            source = self.source_list[batch_idx][sample_idx]
            label = self.label_list[batch_idx][sample_idx]
        else:
            data, frequency, source, label = self.load_every(self.batches[batch_idx])
            data = data[sample_idx]
            frequency = frequency[sample_idx]
            source = source[sample_idx]
            label = label[sample_idx]
        if self.transform_data:
            data = self.transform_data(data)
        if self.transform_frequency:
            frequency = self.transform_frequency(frequency)
        if self.transform_source:
            source = self.transform_source(source)
        if self.transform_label and label is not None:
            label = self.transform_label(label)
        return data, frequency, source, label if label is not None else np.array([])

    def __len__(self):
        return len(self.batches) * self.file_size


if __name__ == '__main__':
    transform_data = Compose([
        T.LogTransform(k=1),
        T.MinMaxNormalize(T.log_transform(-61, k=1), T.log_transform(120, k=1))
    ])
    transform_label = Compose([
        T.MinMaxNormalize(2000, 6000)
    ])
    dataset = FWIDataset(f'relevant_files/temp.txt', transform_data=transform_data, transform_label=transform_label, file_size=1)
    data, label = dataset[0]
    print(data.shape)
    print(label is None)
