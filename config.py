class Env(object):
    env = 'PoetryGen'  # visdom env
    visdom = True  # 是否使用visdom可視化
    batch_size = 1  # Batch Size
    use_gpu = True  # 是否使用GPU加速
    use_teacher_forcing = True  # 是否使用Teacher Forcing
    ratio = 0.6  # training data比例
    batch_size = 10  # 批次訓練數量
    epochs = 100  # 迭代次數
    hidden_size = 5  # hidden unit數量
    encoder_layers = 2  # encoder層數
    dropout = 0.0
