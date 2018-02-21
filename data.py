import os
import pandas as pd
import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm


class ChinsePoetry():
    def __init__(self):
        self.path = 'data/'
        self.fileNames = self._fileNames()

    def _fileNames(self, subFileName='.json', startFileName='poet'):
        path = self.path
        jsonFiles = [file for file in os.listdir(path) if file.endswith(subFileName) and file.startswith(startFileName)]
        return jsonFiles

    def getNumpys(self, msg, progress=True):
        return self.getPandas(msg, progress).values

    def getPandas(self, msg, progress=True):
        data = pd.DataFrame([])
        fileNames = self.fileNames
        if progress:
            print(msg)
            fileNames = tqdm(self.fileNames)

        for fileName in fileNames:
            data = data.append(pd.read_json(self.path + fileName))

        return data


class Poet(data.Dataset):
    def __init__(self, type='train', seed=1):
        data = np.load('data/poet.npz')
        length = data['poetry'].shape[0]
        self.type = type  # train, val, test
        self.length = length

        np.random.seed(seed)  # 讓每次打亂random都一樣
        np.random.shuffle(data['poetry'])

        if type == 'train':
            self.data = data['poetry'][int(0.6 * length):]

    def __getitem__(self, index):
        # TODO: One-Hot encoding
        item = self.data[index]
        return item

    def __len__(self):
        return self.length
