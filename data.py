import os
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm
from config import Env
import pickle

options = Env()


class OneHot(nn.Module):
    def __init__(self, depth):
        super(OneHot, self).__init__()
        self.depth = depth
        self.ones = torch.sparse.torch.eye(depth)

    def forward(self, X_in):
        X_in = X_in.long()
        return Variable(self.ones.index_select(0, X_in.data))

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.depth)


class ChinsePoetry():
    def __init__(self):
        self.path = 'data/'
        self.fileNames = self._fileNames()

    def _fileNames(self, subFileName='.json', startFileName='poet.tang'):
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
    def __init__(self, type='train', ratio=0.6, seed=1):
        filterID = [1003, 2283, 6210, 12120, 12183, 12259, 12353, 12378, 12501, 12804, 13518, 18063, 19086, 19088,
                    20609, 20797, 21120, 25319, 30347, 30356, 31885, 32366, 32432, 32461, 36881, 37581, 40588, 40955,
                    43140, 43148, 44730, 46648, 47140, 48872, 48918, 55525]  # 處理遺漏值

        self.npz = np.delete(np.load('data/poet.npz')['poetry'], filterID, axis=0)
        # self.npz = np.load('data/poet.npz')['poetry']
        self.labelEncoder = pickle.load(open('data/label.pickle', 'rb'))
        self.type = type  # train, val, test
        self.resultLen = 64
        self.dataLen = 8
        self.EOS = len(self.labelEncoder.classes_) - 1
        self.SOS = len(self.labelEncoder.classes_) - 2
        self.PAD = 0

        np.random.seed(seed)  # 讓每次打亂random都一樣
        np.random.shuffle(self.npz)

        if type == 'train':
            myMap = lambda x: x[:int(ratio * self.npz.shape[0]), 1]
            self.data = myMap(self.npz)

    def getWordDim(self):
        encoder = self.labelEncoder
        return len(encoder.classes_)

    def __getitem__(self, index):
        labelEncoder = self.labelEncoder
        resultMap = lambda x: torch.from_numpy(
            np.append(np.append([self.SOS], labelEncoder.transform(list(''.join(x))[0:self.resultLen])), [self.EOS]))
        dataMap = lambda x: torch.from_numpy(
            np.append(np.append([self.SOS], labelEncoder.transform(list(map(lambda xx: xx[0], x))[0:self.dataLen])),
                      [self.EOS]))

        resultItem = resultMap(self.data[index])
        dataItem = dataMap(self.data[index])

        return [dataItem, resultItem]

    def __len__(self):
        return len(self.data)
