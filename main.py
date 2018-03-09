from utils import Visualizer
from data import ChinsePoetry, Poet
from tqdm import tqdm
from config import Env
from model import Encoder, Decoder
from torch.autograd import Variable
from torch.utils import data as D
from func import padSeq, getSeqLength, emap, lmap
from loss import masked_cross_entropy
import torch.nn as nn
import numpy as np
import torch
import multiprocessing as mp

options = Env()


def train(**kwargs):
    for k, v in kwargs.items():
        setattr(options, k, v)

    vis = Visualizer(env=options.env)
    poet = Poet(type='train', ratio=options.ratio)
    [maxEnLength, maxDeLength] = [poet.dataLen + 2, poet.resultLen + 2]

    print('padding sequences...')
    [enPadded, dePadded] = lmap(lambda x, y: torch.zeros(len(x), y), [poet, poet], [maxEnLength, maxDeLength])
    for i, [data, result] in tqdm(enumerate(poet), total=len(poet)):
        # TODO: 有沒有方法可以加速?
        [enPadded[i], dePadded[i]] = lmap(lambda x, y, z: padSeq(x, y, z), [data, result], [maxEnLength, maxDeLength],
                                          [poet.PAD, poet.PAD])

    [enTensor, deTensor] = lmap(torch.stack, [enPadded, dePadded])
    dataset = D.TensorDataset(data_tensor=enTensor.long(), target_tensor=deTensor.long())
    loader = D.DataLoader(dataset=dataset, batch_size=options.batch_size, drop_last=True, num_workers=0)

    encoder = Encoder.RNN(poet.getWordDim(), options.hidden_size, options.encoder_layers, options.dropout)
    decoder = Decoder.LAttnRNN('general', options.hidden_size, poet.getWordDim())
    encoder.cuda() if options.use_gpu else None
    decoder.cuda() if options.use_gpu else None

    print('Training...')
    for epoch in tqdm(range(options.epochs)):
        totalLoss = 0

        for batchX, batchY in tqdm(loader):
            # 找出batch中每個sequence的長度
            [batchX, batchY] = lmap(lambda x: x.tolist(), [batchX, batchY])
            [lenX, lenY] = lmap(lambda x, y: [s.index(x) + 1 for s in y], [poet.EOS, poet.EOS], [batchX, batchY])

            # 將batch中sequences按長到短進行排序
            sortBatch = lambda x, y: zip(*sorted(zip(x, y), key=lambda x: x[0], reverse=True))
            [(lenX, batchX), (lenY, batchY)] = lmap(sortBatch, [lenX, lenY], [batchX, batchY])

            # 存成Variable
            [varX, varY] = lmap(lambda x: Variable(torch.Tensor(x).long()).transpose(0, 1),
                                [batchX, batchY])

            [varX, varY] = lmap(lambda x: x.cuda() if options.use_gpu else x, [varX, varY])

            # 輸入encoder
            enOuts, enHidden = encoder(varX, list(lenX), None)

            # 準備decoder的輸入
            deIn = Variable(torch.Tensor([poet.SOS] * options.batch_size).long())
            allDeOuts = Variable(torch.zeros(max(lenY), options.batch_size, decoder.output_size))

            [deIn, allDeOuts] = lmap(lambda x: x.cuda() if options.use_gpu else x, [deIn, allDeOuts])

            # 輸入decoder
            deHidden = enHidden[:decoder.n_layers]
            loss = 0

            for t in tqdm(range(max(lenY))):
                deOut, deHidden, deAttn = decoder(deIn, deHidden, enOuts, options.use_gpu)
                allDeOuts[t] = deOut
                deIn = varY[t]  # 下一次的輸入是這一次的輸出
                loss += nn.NLLLoss()(nn.LogSoftmax()(deOut), varY[t])

            loss.backward()
            totalLoss += loss / (varY.size()[0] / options.batch_size)

        torch.save({
            'epoch': str(epoch + 1),
            'loss': str(totalLoss.data[0]),
            'batch_size': options.batch_size,
            'enState': encoder.state_dict(),
            'deState': decoder.state_dict()
        }, 'model-loss-' + str(int(totalLoss.data[0])) + '.pt')

        tqdm.write('epoch = ' + str(epoch + 1) + ', loss = ' + str(totalLoss.data[0]))


def saveNpz(**kwargs):
    poetry = ChinsePoetry()
    data = poetry.getNumpys(msg='將資料轉為Numpy...')
    np.savez_compressed('data/poet.npz', poetry=data)


if __name__ == '__main__':
    import fire

    fire.Fire()
