import os
import numpy as np
import pandas as pd
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
        return self.getPandas(msg).values

    def getPandas(self, msg, progress=True):
        data = pd.DataFrame([])
        fileNames = self.fileNames
        if progress:
            print(msg)
            fileNames = tqdm(self.fileNames)

        for fileName in fileNames:
            data = data.append(pd.read_json(self.path + fileName))

        return data

    def getFileNames(self):
        return self.fileNames
