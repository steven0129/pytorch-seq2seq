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


def train(**kwargs):
    for k, v in kwargs.items():
        setattr(options, k, v)

    # 創建新visdom
    vis = Visualizer(env=options.env)

    # 拿取data
    poet = Poet(type='train', ratio=options.ratio)

    # 求seq最大長度
    encoderLengths = torch.zeros(len(poet)).type(torch.LongTensor)
    decoderLengths = torch.zeros(len(poet)).type(torch.LongTensor)

    print('正在計算seq最大長度...')
    for i, [encoderLength, decoderLength] in tqdm(enumerate(map(getSeqLength, poet)), total=len(poet)):
        # TODO: 有沒有方法可以加速?
        encoderLengths[i] = encoderLength
        decoderLengths[i] = decoderLength

    maxDecoderLength = torch.max(decoderLengths)  # decoder最大長度
    maxEncoderLength = torch.max(encoderLengths)  # encoder最大長度

    # 幫sequence補0
    encoderPadded = torch.zeros(len(poet), maxEncoderLength)
    decoderPadded = torch.zeros(len(poet), maxDecoderLength)

    print('padding sequences...')
    for i, [data, result] in tqdm(enumerate(poet), total=len(poet)):
        # TODO: 有沒有方法可以加速?
        encoderPadded[i] = padSeq(data, maxEncoderLength, poet.PAD)
        decoderPadded[i] = padSeq(result, maxDecoderLength, poet.PAD)

    # TODO: 訓練Seq2seq Model
    encoderTensor = torch.stack(encoderPadded)
    decoderTensor = torch.stack(decoderPadded)
    dataset = D.TensorDataset(data_tensor=encoderTensor.type(torch.LongTensor),
                              target_tensor=decoderTensor.type(torch.LongTensor))
    loader = D.DataLoader(dataset=dataset, batch_size=options.batch_size, num_workers=options.CPU)

    # Encoder with GRU
    encoder = Encoder.RNN(poet.getWordDim(), options.hidden_size, options.encoder_layers, options.dropout)
    encoder.cuda() if options.use_gpu else None

    print('Training...')
    for epoch in tqdm(range(options.epochs)):
        for batchX, batchY in tqdm(loader):
            # 從長到短排序
            batchX = batchX.tolist()
            batchY = batchY.tolist()
            lenX = [s.index(poet.EOS) + 1 for s in batchX]
            lenY = [s.index(poet.EOS) + 1 for s in batchY]

            pairMap = lambda len, batch: sorted(zip(len, batch), key=lambda x: x[0], reverse=True)
            pairsX = pairMap(lenX, batchX)
            pairsY = pairMap(lenY, batchY)
            lenX, batchX = zip(*pairsX)
            lenY, batchY = zip(*pairsY)

            # 輸入encoder
            varX = Variable(torch.LongTensor(batchX)).transpose(0, 1).contiguous()
            varY = Variable(torch.LongTensor(batchY)).transpose(0, 1).contiguous()

            varX = varX.cuda() if options.use_gpu else varX
            varY = varY.cuda() if options.use_gpu else varY

            encoderOut, encoderHidden = encoder(varX, list(lenX), None)


def saveNpz(**kwargs):
    poetry = ChinsePoetry()
    data = poetry.getNumpys(msg='將資料轉為Numpy...')
    np.savez_compressed('data/poet.npz', poetry=data)


if __name__ == '__main__':
    import fire

    fire.Fire()
