from utils import Visualizer
from data import ChinsePoetry
from data import Poet
from tqdm import tqdm
from config import Env
import numpy as np
import torch

options = Env()


def train(**kwargs):
    for k, v in kwargs.items():
        setattr(options, k, v)

    # 創建新visdom
    vis = Visualizer(env=options.env)

    # 拿取data
    poet = Poet(type='train')
    # print(poet[131628])

    # 求seq最大長度
    decoderLengths = []
    encoderLengths = []
    print('正在計算seq最大長度...')

    with tqdm(total=len(poet)) as pbar:
        for [data, result] in poet:
            getSeqLength = lambda x: list(x.size())[0]
            encoderLengths.append(getSeqLength(data))
            decoderLengths.append(getSeqLength(result))
            pbar.update()

    maxDecoderLength = max(decoderLengths)  # decoder最大長度
    maxEncoderLength = max(encoderLengths)  # encoder最大長度
    print('decoder最大長度為: ' + str(maxDecoderLength))
    print('encoder最大長度為: ' + str(maxEncoderLength))

    # TODO: 搭建Seq2seq Model


def saveNpz(**kwargs):
    poetry = ChinsePoetry()
    data = poetry.getNumpys(msg='將資料轉為Numpy...')
    np.savez_compressed('data/poet.npz', poetry=data)


if __name__ == '__main__':
    import fire

    fire.Fire()
