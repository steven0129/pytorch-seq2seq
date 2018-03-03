import torch.nn.functional as F

def padSeq(tensor, max_length, symbol):
    # Pad a with the PAD symbol
    tensor = F.pad(tensor, (0, max_length - tensor.size()[0]), 'constant', symbol).data
    return tensor


def getSeqLength(arr):
    myMap = lambda x: list(map(lambda xx: list(xx.size())[0], x))
    return myMap(arr)


def emap(*args):
    return enumerate(map(*args))


def lmap(*args):
    return list(map(*args))