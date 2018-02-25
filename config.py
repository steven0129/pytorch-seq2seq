class Env(object):
    env = 'PoetryGen'  # visdom env
    visdom = True  # 是否使用visdom可視化
    batch_size = 1  # Batch Size
    use_gpu = True  # 是否使用GPU加速
    use_teacher_forcing = True  # 是否使用Teacher Forcing
    word_dim = 21589
    ratio = 0.6  # training data比例
