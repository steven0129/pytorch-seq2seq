from utils import Visualizer
from data import ChinsePoetry
from data import Poet
from torch.utils.data import DataLoader
import numpy as np


class Config(object):
    env = 'PoetryGen'  # visdom env
    visdom = True  # 是否使用visdom可視化
    batch_size = 1  # Batch Size
    use_gpu = True  # 是否使用GPU加速
    num_workers = 12


options = Config()


def train(**kwargs):
    for k, v in kwargs.items():
        setattr(options, k, v)

    # 創建新visdom
    vis = Visualizer(env=options.env)

    # 拿取data
    # data = np.load('data/poet.npz')
    # print(data['poetry'])
    trainDataloader = DataLoader(dataset=Poet(type='train'), batch_size=options.batch_size, num_workers=options.num_workers)

    # TODO: 搭建Seq2seq Model
    for data in enumerate(trainDataloader):
        print(data)


def saveNpz(**kwargs):
    poetry = ChinsePoetry()
    data = poetry.getNumpys(msg='將資料轉為Numpy...')
    np.savez_compressed('data/poet.npz', poetry=data)


if __name__ == '__main__':
    import fire

    fire.Fire()
