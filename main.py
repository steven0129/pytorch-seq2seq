from utils import Visualizer
from data import ChinsePoetry, Poet, OneHot
from tqdm import tqdm
from config import Env
from model import EncoderRNN
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn.functional as F

options = Env()


# Pad a with the PAD symbol
def padSeq(tensor, max_length):
    tensor = F.pad(tensor, (0, max_length - tensor.size()[0]), 'constant', 0)
    return tensor


def train(**kwargs):
    for k, v in kwargs.items():
        setattr(options, k, v)

    # 創建新visdom
    vis = Visualizer(env=options.env)

    # 拿取data
    poet = Poet(type='train')

    # 求seq最大長度
    encoderLengths = torch.zeros(len(poet)).type(torch.LongTensor)
    encoderLengths = encoderLengths.cuda() if options.use_gpu else encoderLengths
    decoderLengths = torch.zeros(len(poet)).type(torch.LongTensor)
    decoderLengths = decoderLengths.cuda() if options.use_gpu else decoderLengths

    print('正在計算seq最大長度...')
    getSeqLength = lambda x: list(x.size())[0]
    for i, [data, result] in tqdm(enumerate(poet), total=len(poet)):
        # TODO: 有沒有方法可以加速?
        encoderLengths[i] = getSeqLength(data)
        decoderLengths[i] = getSeqLength(result)

    maxDecoderLength = torch.max(decoderLengths)  # decoder最大長度
    maxEncoderLength = torch.max(encoderLengths)  # encoder最大長度

    # padding sequeces
    encoderPadded = []
    decoderPadded = []

    print('padding sequences...')
    for i, [data, result] in tqdm(enumerate(poet), total=len(poet)):
        # TODO: 有沒有方法可以加速?
        encoderPadded.append(padSeq(data, maxEncoderLength))
        decoderPadded.append(padSeq(result, maxDecoderLength))

    print('encoder最大長度為: ' + str(maxEncoderLength))
    print('padded encoder shape = ' + str(torch.stack(encoderPadded).shape))
    print('decoder最大長度為: ' + str(maxDecoderLength))
    print('padded decoder shape = ' + str(torch.stack(decoderPadded).shape))

    # TODO: 訓練Seq2seq Model

    encoderVar = torch.stack(encoderPadded).transpose(0, 1)
    decoderVar = torch.stack(decoderPadded).transpose(0, 1)

    print(encoderVar.shape)
    print(decoderVar.shape)


def saveNpz(**kwargs):
    poetry = ChinsePoetry()
    data = poetry.getNumpys(msg='將資料轉為Numpy...')
    np.savez_compressed('data/poet.npz', poetry=data)


if __name__ == '__main__':
    import fire

    fire.Fire()
