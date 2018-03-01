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
    def __init__(self, type='train', ratio=0.6, seed=1):
        filter = [6240, 7802, 12280, 24700, 26395, 31326, 38520, 40163, 50285, 53447, 70357, 70420, 70496, 70590,
                  70615, 70738, 71041, 73598, 74755, 84433, 89316, 95806, 97300, 104323, 104325, 107020, 108846,
                  109034, 109357, 109845, 109914, 109984, 110083, 110085, 110106, 110119, 110355, 110745, 131628,
                  134157, 134252, 134436, 137556, 138876, 146343, 149505, 152865, 155178, 163492, 166584, 166593,
                  179122, 179603, 179669, 179698, 197369, 197739, 197743, 197771, 197802, 200813, 201415, 211829,
                  222786, 225349, 230059, 234159, 234634, 240251, 251118, 251818, 255825, 255917, 257192, 263377,
                  263385, 263912, 263976, 263987, 264959, 273967, 279885, 280377, 286757, 287437, 288109, 288155,
                  302762, 310070, 310072, 310074, 310076, 310078, 310080, 310082, 310084, 310086, 310088, 310090,
                  310092, 310094, 310096, 310098, 310100, 310102, 310104, 310106, 310108]  # 遺漏值索引
        self.npz = np.delete(np.load('data/poet.npz')['poetry'], filter, axis=0)
        self.labelEncoder = pickle.load(open('data/label.pickle', 'rb'))
        self.type = type  # train, val, test
        self.EOS = options.word_dim + 1
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
        resultMap = lambda x: torch.from_numpy(np.append(labelEncoder.transform(list(''.join(x))), [self.EOS]))
        dataMap = lambda x: torch.from_numpy(
            np.append(labelEncoder.transform(list(map(lambda x: x[0], x))), [self.EOS]))

        resultItem = resultMap(self.data[index])
        dataItem = dataMap(self.data[index])

        return [dataItem, resultItem]

    def __len__(self):
        return len(self.data)
