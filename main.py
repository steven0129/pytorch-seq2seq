from utils import Visualizer
from data import ChinsePoetry
from data import Poet
from tqdm import tqdm
import numpy as np
import multiprocessing as mp
import torch


class Config(object):
    env = 'PoetryGen'  # visdom env
    visdom = True  # 是否使用visdom可視化
    batch_size = 1  # Batch Size
    use_gpu = True  # 是否使用GPU加速
    num_workers = 12


options = Config()


def getSeqLength(x):
    return list(x.size())[0]


def train(**kwargs):
    for k, v in kwargs.items():
        setattr(options, k, v)

    # 創建新visdom
    vis = Visualizer(env=options.env)

    # 拿取data
    poet = Poet(type='train')

    # 求seq最大長度
    seqLengths=[]
    print('正在計算seq最大長度...')

    with mp.Pool() as pool: # CPU平行化
        with tqdm(total=len(poet)) as pbar:
            for _, item in enumerate(pool.imap_unordered(getSeqLength, poet)): # 不按順序的放入執行緒池
                seqLengths.append(item)
                pbar.update()

    maxSeqLength = max(seqLengths)
    print('訓練集中seq最大長度為: ' + str(maxSeqLength))

    # TODO: 搭建Seq2seq Model


def saveNpz(**kwargs):
    poetry = ChinsePoetry()
    data = poetry.getNumpys(msg='將資料轉為Numpy...')
    np.savez_compressed('data/poet.npz', poetry=data)


if __name__ == '__main__':
    import fire

    fire.Fire()
