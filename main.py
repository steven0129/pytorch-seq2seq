from utils import Visualizer
from data import ChinsePoetry, Poet
from tqdm import tqdm
from config import Env
from model import Encoder
from torch.autograd import Variable
from torch.utils import data as D
from func import padSeq, getSeqLength, emap, lmap
import numpy as np
import torch

options = Env()


def train(**kwargs):
    for k, v in kwargs.items():
        setattr(options, k, v)

    vis = Visualizer(env=options.env)
    poet = Poet(type='train', ratio=options.ratio)
    [enLengths, deLengths] = lmap(lambda x: torch.zeros(len(x)).long(), [poet, poet])

    print('正在計算seq最大長度...')
    for i, [enLength, deLength] in tqdm(emap(getSeqLength, poet), total=len(poet)):
        # TODO: 有沒有方法可以加速?
        [enLengths[i], deLengths[i]] = lmap(lambda x: x, [enLength, deLength])

    [maxEnLength, maxDeLength] = lmap(torch.max, [enLengths, deLengths])  # encoder, decoder最大長度

    print('padding sequences...')
    [enPadded, dePadded] = lmap(lambda x, y: torch.zeros(len(x), y), [poet, poet],
                                          [maxEnLength, maxDeLength])
    for i, [data, result] in tqdm(enumerate(poet), total=len(poet)):
        # TODO: 有沒有方法可以加速?
        [enPadded[i], dePadded[i]] = lmap(lambda x, y, z: padSeq(x, y, z), [data, result],
                                                    [maxEnLength, maxDeLength], [poet.PAD, poet.PAD])

    [enTensor, deTensor] = lmap(torch.stack, [enPadded, dePadded])
    dataset = D.TensorDataset(data_tensor=enTensor.long(), target_tensor=deTensor.long())
    loader = D.DataLoader(dataset=dataset, batch_size=options.batch_size, num_workers=options.CPU)

    encoder = Encoder.RNN(poet.getWordDim(), options.hidden_size, options.encoder_layers, options.dropout)
    encoder.cuda() if options.use_gpu else None

    print('Training...')
    for epoch in tqdm(range(options.epochs)):
        for batchX, batchY in tqdm(loader):
            # 找出batch中每個sequence的長度
            [batchX, batchY] = lmap(lambda x: x.tolist(), [batchX, batchY])
            [lenX, lenY] = lmap(lambda x, y: [s.index(x) + 1 for s in y], [poet.EOS, poet.EOS], [batchX, batchY])

            # 將batch中sequences按長到短進行排序
            sortBatch = lambda x, y: zip(*sorted(zip(x, y), key=lambda x: x[0], reverse=True))
            [(lenX, batchX), (lenY, batchY)] = lmap(sortBatch, [lenX, lenY], [batchX, batchY])

            # 存成Variable
            [varX, varY] = lmap(lambda x: Variable(torch.LongTensor(x)).transpose(0, 1).contiguous(), [batchX, batchY])
            [varX, varY] = lmap(lambda x: x.cuda() if options.use_gpu else x, [varX, varY])

            # 輸入encoder
            enOut, enHidden = encoder(varX, list(lenX), None)
            tqdm.write(str(enHidden.size()))


def saveNpz(**kwargs):
    poetry = ChinsePoetry()
    data = poetry.getNumpys(msg='將資料轉為Numpy...')
    np.savez_compressed('data/poet.npz', poetry=data)


if __name__ == '__main__':
    import fire

    fire.Fire()
