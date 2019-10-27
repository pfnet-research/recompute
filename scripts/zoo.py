# model zoo

import chainer
import chainer.links as L
import chainer.functions as F
from chainercv.links import PixelwiseSoftmaxClassifier
from chainercv.links import SegNetBasic
from chainercv.links import FasterRCNNVGG16
from chainercv.links.model.faster_rcnn import FasterRCNNTrainChain

from densenet import DenseNet
from long_rnn import LongLSTM, LongLinear
from unet import UNET

import numpy as np
from pspnet import pspnet
from ssd import ssd300, ssd512


class MLPClassifier(chainer.Chain):

    def __init__(self, n_units, n_out):
        super().__init__()
        with self.init_scope():
            self.l1 = L.Linear(784, n_units)
            self.l2 = L.Linear(n_units, n_out)

    def forward(self, x, t):
        h1 = F.relu(self.l1(x))
        h2 = self.l2(h1)
        y = F.softmax_cross_entropy(h2, t)
        return y


class DiamondClassifier(chainer.Chain):

    def __init__(self, n_units, n_out):
        super().__init__()
        with self.init_scope():
            self.l1 = L.Linear(784, n_units)
            self.l2 = L.Linear(n_units, n_out)
            self.l3 = L.Linear(n_units, n_out)

    def forward(self, x, t):
        h1 = self.l1(x)
        h2 = self.l2(h1)
        h3 = self.l3(h1)
        h4 = h2 + h3
        y = F.softmax_cross_entropy(h4, t)
        return y


class AveClassifier(chainer.Chain):
    # nonsense model

    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, 10)

    def forward(self, x, t):
        h = F.average_pooling_2d(x, 2, 2)
        h = self.l1(h)
        y = F.softmax_cross_entropy(h, t)
        return y


class ConvReLUClassifier(chainer.Chain):
    # nonsense model

    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.l1 = L.Linear(10000, 10000)

    def forward(self, x, t):
        for _ in range(20):
            x = F.relu(self.l1(x))
        y = F.softmax_cross_entropy(x, t)
        return y


def get_mlp(batchsize):
    model = MLPClassifier(784, 10)
    x = np.random.uniform(size=(batchsize, 784)).astype('f')
    x = chainer.as_variable(x)
    t = np.random.randint(size=(batchsize,), low=0, high=10)\
        .astype(np.int32)
    t = chainer.as_variable(t)
    return [x, t], model


def get_diamond(batchsize):
    model = DiamondClassifier(784, 10)
    x = np.random.uniform(size=(batchsize, 784)).astype('f')
    x = chainer.as_variable(x)
    t = np.random.randint(size=(batchsize,), low=0, high=10)\
        .astype(np.int32)
    t = chainer.as_variable(t)
    return [x, t], model


def get_segnet(batchsize):
    model = SegNetBasic(n_class=17)
    model = PixelwiseSoftmaxClassifier(
        model, class_weight=np.ones(17))
    x = np.random.uniform(size=(batchsize, 3, 1024, 1024)).astype('f')
    x = chainer.as_variable(x)
    t = np.random.randint(size=(batchsize, 1024, 1024), low=0, high=10)\
        .astype(np.int32)
    t = chainer.as_variable(t)
    return [x, t], model


class Wrapper(chainer.Chain):

    def __init__(self, predictor, key=None):
        super().__init__()

        with self.init_scope():
            self.predictor = predictor
        self.key = key

    def __call__(self, x, t):
        if self.key is None:
            y = self.predictor(x)
        else:
            y = self.predictor(x, layers=[self.key])[self.key]
        y = F.softmax_cross_entropy(y, t)
        return y


def get_resnet50(batchsize):
    model = L.ResNet50Layers(pretrained_model=None)
    model = Wrapper(model, 'fc6')
    x = np.random.uniform(size=(batchsize, 3, 224, 224)).astype('f')
    x = chainer.as_variable(x)
    t = np.random.randint(size=(batchsize,), low=0, high=1000).astype(np.int32)
    t = chainer.as_variable(t)
    return [x, t], model


def get_resnet152(batchsize):
    model = L.ResNet152Layers(pretrained_model=None)
    model = Wrapper(model, 'fc6')
    x = np.random.uniform(size=(batchsize, 3, 224, 224)).astype('f')
    x = chainer.as_variable(x)
    t = np.random.randint(size=(batchsize,), low=0, high=1000).astype(np.int32)
    t = chainer.as_variable(t)
    return [x, t], model


class WrapperGooglenet(chainer.Chain):

    def __init__(self, predictor):
        super().__init__()

        with self.init_scope():
            self.predictor = predictor

    def __call__(self, x, t1, t2, t3):
        y = self.predictor(x, layers=['loss3_fc', 'loss1_fc2', 'loss2_fc2'])
        loss1 = F.softmax_cross_entropy(y['loss1_fc2'], t1)
        loss2 = F.softmax_cross_entropy(y['loss2_fc2'], t2)
        loss3 = F.softmax_cross_entropy(y['loss3_fc'], t3)
        y = loss1 + loss2 + loss3
        return y


def get_googlenet(batchsize):
    model = L.GoogLeNet(pretrained_model=None)
    model = WrapperGooglenet(model)
    x = np.random.uniform(size=(batchsize, 3, 224, 224)).astype('f')
    x = chainer.as_variable(x)
    t1 = np.random.randint(size=(batchsize,), low=0, high=1000)
    t1 = chainer.as_variable(t1.astype(np.int32))
    t2 = np.random.randint(size=(batchsize,), low=0, high=1000)
    t2 = chainer.as_variable(t2.astype(np.int32))
    t3 = np.random.randint(size=(batchsize,), low=0, high=1000)
    t3 = chainer.as_variable(t3.astype(np.int32))

    return [x, t1, t2, t3], model


def get_vgg16(batchsize):
    model = L.VGG16Layers(pretrained_model=None)
    model = Wrapper(model, 'fc8')
    x = np.random.uniform(size=(batchsize, 3, 224, 224)).astype('f')
    x = chainer.as_variable(x)
    t = np.random.randint(size=(batchsize,), low=0, high=1000).astype(np.int32)
    t = chainer.as_variable(t)

    return [x, t], model


def get_vgg19(batchsize):
    model = L.VGG19Layers(pretrained_model=None)
    model = Wrapper(model, 'fc8')
    x = np.random.uniform(size=(batchsize, 3, 224, 224)).astype('f')
    x = chainer.as_variable(x)
    t = np.random.randint(size=(batchsize,), low=0, high=1000).astype(np.int32)
    t = chainer.as_variable(t)

    return [x, t], model


def get_densenet(batchsize):
    model = DenseNet(161)
    model = Wrapper(model)
    x = np.random.uniform(size=(batchsize, 3, 224, 224)).astype('f')
    x = chainer.as_variable(x)
    t = np.random.randint(size=(batchsize,), low=0, high=1000).astype(np.int32)
    t = chainer.as_variable(t)

    return [x, t], model


def get_lstm(batchsize):
    model = LongLSTM()
    xs = [np.random.uniform(size=(batchsize, 200)).astype('f')
          for _ in range(256)]
    xs = [chainer.as_variable(x) for x in xs]
    c0 = np.random.uniform(size=(batchsize, 50)).astype('f')
    c0 = chainer.as_variable(c0)
    t = np.random.uniform(size=(batchsize, 50)).astype('f')
    t = chainer.as_variable(t)
    xs.extend([c0, t])

    return xs, model


def get_longlinear(batchsize):
    model = LongLinear()
    x = np.random.uniform(size=(batchsize, 1000)).astype('f')
    x = chainer.as_variable(x)
    t = np.random.uniform(size=(batchsize, 1000)).astype('f')
    t = chainer.as_variable(t)
    return [x, t], model


def get_unet(batchsize):
    model = UNET()
    x = np.random.uniform(size=(batchsize, 1, 572, 572)).astype('f')
    x = chainer.as_variable(x)
    t = np.random.randint(size=(batchsize, 388, 388),
                          low=0, high=2).astype(np.int32)
    t = chainer.as_variable(t)

    return [x, t], model


def get_pspnet(batchsize):
    model = pspnet()
    x = np.random.uniform(size=(batchsize, 3, 713, 713)).astype('f')
    x = chainer.as_variable(x)
    t = np.random.randint(size=(batchsize, 713, 713,), low=0, high=10)\
        .astype(np.int32)
    t = chainer.as_variable(t)

    return [x, t], model


def get_ssd300(batchsize):
    model = ssd300()
    K = 24656
    x = np.random.uniform(size=(batchsize, 3, 512, 512)).astype('f')
    x = chainer.as_variable(x)
    gt_mb_locs = np.random.uniform(size=(batchsize, K, 4)).astype('f')
    gt_mb_locs = chainer.as_variable(gt_mb_locs)
    gt_mb_labels = np.random.randint(size=(batchsize, K), low=0, high=20)\
        .astype(np.int32)
    gt_mb_labels = chainer.as_variable(gt_mb_labels)

    return [x, gt_mb_locs, gt_mb_labels], model


def get_ssd512(batchsize):
    model = ssd512()
    K = 24564
    x = np.random.uniform(size=(batchsize, 3, 512, 512)).astype('f')
    x = chainer.as_variable(x)
    gt_mb_locs = np.random.uniform(size=(batchsize, K, 4)).astype('f')
    gt_mb_locs = chainer.as_variable(gt_mb_locs)
    gt_mb_labels = np.random.randint(size=(batchsize, K), low=0, high=20)\
        .astype(np.int32)
    gt_mb_labels = chainer.as_variable(gt_mb_labels)

    return [x, gt_mb_locs, gt_mb_labels], model


def get_faster_rcnn(n):
    frc = FasterRCNNVGG16(n_fg_class=20)
    model = FasterRCNNTrainChain(frc)

    batchsize = 1  # only 1 is supported
    K = 10
    x = np.random.uniform(size=(batchsize, 3, n * 512, 512)).astype('f')
    x = chainer.as_variable(x)
    bbox = np.random.uniform(size=(batchsize, K, 4)).astype('f')
    bbox = chainer.as_variable(bbox)
    labels = np.random.randint(size=(batchsize, K), low=0, high=20)\
        .astype(np.int32)
    labels = chainer.as_variable(labels)
    scale = np.ones((batchsize,)).astype('f')
    scale = chainer.as_variable(scale)

    return [x, bbox, labels, scale], model


def get_convrelu(batchsize):
    model = ConvReLUClassifier()
    x = np.random.uniform(size=(batchsize, 10000)).astype('f')
    x = chainer.as_variable(x)
    t = np.random.randint(size=(batchsize,), low=0, high=10).astype(np.int32)
    t = chainer.as_variable(t)

    return [x, t], model


zoo_list = {
    'mlp': (get_mlp, 100),
    'diamond': (get_diamond, 100),
    # 'segnet': get_segnet,
    'resnet50': (get_resnet50, 32),
    'resnet152': (get_resnet152, 32),
    'googlenet': (get_googlenet, 32),
    'vgg16': (get_vgg16, 32),
    'vgg19': (get_vgg19, 32),
    'densenet': (get_densenet, 32),
    # 'lstm': get_lstm,
    # 'longlinear': get_longlinear,
    'unet': (get_unet, 8),
    'pspnet': (get_pspnet, 1),
    'ssd300': (get_ssd300, 4),
    'ssd512': (get_ssd512, 4),
    'faster_rcnn': (get_faster_rcnn, 4),
    'convrelu': (get_convrelu, 4000),
}
