from chainercv.experimental.links import PSPNetResNet101
import chainer
import chainer.functions as F
import chainer.links as L


from chainercv.links import Conv2DBNActiv


class TrainChain(chainer.Chain):

    def __init__(self, model, ignore_label=-1):
        initialW = chainer.initializers.HeNormal()
        super(TrainChain, self).__init__()
        with self.init_scope():
            self.model = model
            self.aux_conv1 = Conv2DBNActiv(
                None, 512, 3, 1, 1, initialW=initialW)
            self.aux_conv2 = L.Convolution2D(
                None, model.n_class, 3, 1, 1, False, initialW=initialW)
        self.ignore_label = ignore_label

    def __call__(self, imgs, labels):
        h_aux, h_main = self.model.extractor(imgs)
        h_aux = F.dropout(self.aux_conv1(h_aux), ratio=0.1)
        h_aux = self.aux_conv2(h_aux)
        h_aux = F.resize_images(h_aux, imgs.shape[2:])

        h_main = self.model.ppm(h_main)
        h_main = F.dropout(self.model.head_conv1(h_main), ratio=0.1)
        h_main = self.model.head_conv2(h_main)
        h_main = F.resize_images(h_main, imgs.shape[2:])

        aux_loss = F.softmax_cross_entropy(
            h_aux, labels, ignore_label=self.ignore_label)
        main_loss = F.softmax_cross_entropy(
            h_main, labels, ignore_label=self.ignore_label)
        loss = 0.4 * aux_loss + main_loss

        chainer.reporter.report({'loss': loss}, self)
        return loss


def pspnet():
    model = PSPNetResNet101(17, input_size=(713, 713))
    return TrainChain(model)
