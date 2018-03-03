from utils import Visualizer
from data import ChinsePoetry, Poet
from tqdm import tqdm
from config import Env
from torch.autograd import Variable
from model import Encoder
import numpy as np
import torch
from torch.utils import data as D
import torch.nn.functional as F

options = Env()


# Pad a with the PAD symbol
def padSeq(tensor, max_length, symbol):
    tensor = F.pad(tensor, (0, max_length - tensor.size()[0]), 'constant', symbol).data
    return tensor


def getSeqLength(arr):
    myMap = lambda x: list(map(lambda xx: list(xx.size())[0], x))
    return myMap(arr)


def emap(*args):
    return enumerate(map(*args))


def lmap(*args):
    return list(map(*args))


def train(**kwargs):
    lmap(lambda x, y: x + y, [1], [2])

    for k, v in kwargs.items():
        setattr(options, k, v)

    # 創建新visdom
    vis = Visualizer(env=options.env)

    # 拿取data
    poet = Poet(type='train', ratio=options.ratio)

    # 求seq最大長度
    [encoderLengths, decoderLengths] = lmap(lambda x: torch.zeros(len(x)).long(), [poet, poet])

    print('正在計算seq最大長度...')
    for i, [encoderLength, decoderLength] in tqdm(emap(getSeqLength, poet), total=len(poet)):
        # TODO: 有沒有方法可以加速?
        [encoderLengths[i], decoderLengths[i]] = lmap(lambda x: x, [encoderLength, decoderLength])

    [maxEncoderLength, maxDecoderLength] = lmap(torch.max, [encoderLengths, decoderLengths])  # encoder, decoder最大長度

    # 幫sequence補0
    print('padding sequences...')
    [encoderPadded, decoderPadded] = lmap(lambda x, y: torch.zeros(len(x), y), [poet, poet],
                                          [maxEncoderLength, maxDecoderLength])
    for i, [data, result] in tqdm(enumerate(poet), total=len(poet)):
        # TODO: 有沒有方法可以加速?
        [encoderPadded[i], decoderPadded[i]] = lmap(lambda x, y, z: padSeq(x, y, z), [data, result],
                                                    [maxEncoderLength, maxDecoderLength], [poet.PAD, poet.PAD])

    # 訓練Seq2seq Model
    [encoderTensor, decoderTensor] = lmap(torch.stack, [encoderPadded, decoderPadded])
    dataset = D.TensorDataset(data_tensor=encoderTensor.long(), target_tensor=decoderTensor.long())
    loader = D.DataLoader(dataset=dataset, batch_size=options.batch_size, num_workers=options.CPU)

    # Encoder with GRU
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

            #  存成Variable
            [varX, varY] = lmap(lambda x: Variable(torch.LongTensor(x)).transpose(0, 1).contiguous(), [batchX, batchY])
            [varX, varY] = lmap(lambda x: x.cuda() if options.use_gpu else x, [varX, varY])

            # 輸入encoder
            encoderOut, encoderHidden = encoder(varX, list(lenX), None)
            tqdm.write(str(encoderHidden.size()))


def saveNpz(**kwargs):
    poetry = ChinsePoetry()
    data = poetry.getNumpys(msg='將資料轉為Numpy...')
    np.savez_compressed('data/poet.npz', poetry=data)


if __name__ == '__main__':
    import fire

    fire.Fire()
