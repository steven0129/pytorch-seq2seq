import os
import pandas as pd
import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm
import pickle


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
        self.npz = np.load('data/poet.npz')['poetry']
        self.type = type  # train, val, test
        self.length = self.npz.shape[0]
        self.word_dim = 21585

        np.random.seed(seed)  # 讓每次打亂random都一樣
        np.random.shuffle(self.npz)

        if type == 'train':
            print('開始讀入資料...')
            self.data = self.npz[:, 1]

    def __getitem__(self, index):
        f = open('data/label.pickle', 'rb')
        labelEncoder = pickle.load(f)
        myData = '\\n'.join(self.data[index])
        myMap = lambda x: torch.from_numpy(labelEncoder.transform(list(x)))
        item = myMap(myData)

        return item

    def __len__(self):
        return self.length
