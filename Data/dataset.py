import torch
from torch import nn
import torch.nn as nn
import numpy as np
import argparse
from torch.utils.data import DataLoader
import torch.nn.parallel.data_parallel as parallel
import torch.optim as optim
import os
import time
import scipy.io
from tensorboardX import SummaryWriter
import scipy.io
class dataset(Dataset):

    def __init__(self , eeg, label, size, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.eeg = eeg
        self.label = label
        self.transform = transform
        self.size = size

    def __len__(self):
        return len(self.size)

    def __getitem__(self, idx):
        
        pathLabel = os.path.join(self.label, str(idx),'.mat')
        pathEEG = os.path.join(self.eeg, str(idx), '.mat')
        label = scipy.io.loadmat(pathLabel)
        label = label['all_label']
        label = torch.tensor(label, dtype=torch.cdouble)
        eeg = scipy.io.loadmat(pathEEG)
        eeg = eeg['result']
        eeg = torch.tensor(eeg, dtype=torch.cdouble)
        sample = {'label': label.t(), 'eeg': eeg.t()}

        return sample
        
