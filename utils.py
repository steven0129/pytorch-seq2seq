import numpy as np
import visdom

class Visualizer():
    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.index = {}
        self.log_text = ''

    def plot(self, name, y):
        '''
        :param name: 折線圖名稱
        :param y: 值
        :return:
        '''
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]), win=name, opts=dict(title=name),
                      update=None if x == 0 else 'append')

        self.index[name] = x + 1

    def __getattr__(self, name):
        return getattr(self.vis, name)
