import chainer
import chainer.functions as F
import chainer.links as L


class Bottleneck(chainer.Chain):

    def __init__(self, in_channel, growth_rate):
        super(Bottleneck, self).__init__()

        with self.init_scope():
            mid_channel = 4 * growth_rate

            self.bn0 = L.BatchNormalization(in_channel)
            self.c0 = L.Convolution2D(in_channel, mid_channel, ksize=1, stride=1, pad=0, nobias=True)

            self.bn1 = L.BatchNormalization(mid_channel)
            self.c1 = L.Convolution2D(mid_channel, growth_rate, ksize=3, stride=1, pad=1, nobias=True)

    def __call__(self, x):
        h = self.c0(F.relu(self.bn0(x)))
        h = self.c1(F.relu(self.bn1(h)))

        return F.concat([x, h])


class DenseBlock(chainer.ChainList):

    def __init__(self, in_channel, growth_rate, stage):
        super(DenseBlock, self).__init__()

        for i in range(stage):
            c = in_channel + i * growth_rate
            self.add_link(Bottleneck(c, growth_rate))

        self.out_channel = in_channel + stage * growth_rate

    def __call__(self, x):
        for f in self:
            x = f(x)
        return x


class Transition(chainer.Chain):

    def __init__(self, in_channel, reduction):
        super(Transition, self).__init__()

        self.out_channel = int(reduction * in_channel)

        with self.init_scope():
            self.bn0 = L.BatchNormalization(in_channel)
            self.c0 = L.Convolution2D(in_channel, self.out_channel, ksize=1, stride=1, pad=0, nobias=True)

    def __call__(self, x):
        h = self.c0(F.relu(self.bn0(x)))
        return F.average_pooling_2d(h, 2)


class DenseNet(chainer.Chain):

    def __init__(self, depth, reduction=0.5):
        super(DenseNet, self).__init__()

        if depth == 121:
            stages = (6, 12, 24, 16)
            growth_rate = 32
        elif depth == 169:
            stages = (6, 12, 32, 32)
            growth_rate = 32
        elif depth == 201:
            stages = (6, 12, 48, 32)
            growth_rate = 32
        elif depth == 161:
            stages = (6, 12, 36, 24)
            growth_rate = 48
        else:
            raise

        with self.init_scope():
            channel = 2 * growth_rate
            self.c0 = L.Convolution2D(3, channel, ksize=7, stride=2, pad=3, nobias=True)
            self.bn0 = L.BatchNormalization(channel)

            self.d0 = DenseBlock(channel, growth_rate, stages[0])
            channel = self.d0.out_channel
            self.t0 = Transition(channel, reduction)
            channel = self.t0.out_channel

            self.d1 = DenseBlock(channel, growth_rate, stages[1])
            channel = self.d1.out_channel
            self.t1 = Transition(channel, reduction)
            channel = self.t1.out_channel

            self.d2 = DenseBlock(channel, growth_rate, stages[2])
            channel = self.d2.out_channel
            self.t2 = Transition(channel, reduction)
            channel = self.t2.out_channel

            self.d3 = DenseBlock(channel, growth_rate, stages[3])
            channel = self.d3.out_channel
            self.t3_bn0 = L.BatchNormalization(channel)

            self.fc = L.Linear(channel, 1000)

    def __call__(self, x):
        h = F.relu(self.bn0(self.c0(x)))
        h = F.max_pooling_2d(h, ksize=3, stride=2, pad=1, cover_all=False)

        h = self.d0(h)
        h = self.t0(h)

        h = self.d1(h)
        h = self.t1(h)

        h = self.d2(h)
        h = self.t2(h)

        h = self.d3(h)
        h = F.relu(self.t3_bn0(h))

        h = F.average_pooling_2d(h, h.shape[2:], stride=1)
        h = F.reshape(h, h.shape[:2])

        h = self.fc(h)

        return h
