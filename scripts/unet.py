import chainer
import chainer.functions as F
import chainer.links as L


class UNET(chainer.Chain):

    def __init__(self):
        super(UNET, self).__init__()

        with self.init_scope():
            self.conv_d0a_b = L.Convolution2D(1, 64, ksize=3, stride=1, pad=0, nobias=False)
            self.conv_d0b_c = L.Convolution2D(64, 64, ksize=3, stride=1, pad=0, nobias=False)

            self.conv_d1a_b = L.Convolution2D(64, 128, ksize=3, stride=1, pad=0, nobias=False)
            self.conv_d1b_c = L.Convolution2D(128, 128, ksize=3, stride=1, pad=0, nobias=False)

            self.conv_d2a_b = L.Convolution2D(128, 256, ksize=3, stride=1, pad=0, nobias=False)
            self.conv_d2b_c = L.Convolution2D(256, 256, ksize=3, stride=1, pad=0, nobias=False)

            self.conv_d3a_b = L.Convolution2D(256, 512, ksize=3, stride=1, pad=0, nobias=False)
            self.conv_d3b_c = L.Convolution2D(512, 512, ksize=3, stride=1, pad=0, nobias=False)

            self.conv_d4a_b = L.Convolution2D(512, 1024, ksize=3, stride=1, pad=0, nobias=False)
            self.conv_d4b_c = L.Convolution2D(1024, 1024, ksize=3, stride=1, pad=0, nobias=False)

            self.upconv_d4c_u3a = L.Deconvolution2D(1024, 512, ksize=2, stride=2, pad=0, nobias=False)

            self.conv_u3b_c = L.Convolution2D(1024, 512, ksize=3, stride=1, pad=0, nobias=False)
            self.conv_u3c_d = L.Convolution2D(512, 512, ksize=3, stride=1, pad=0, nobias=False)

            self.upconv_u3d_u2a = L.Deconvolution2D(512, 256, ksize=2, stride=2, pad=0, nobias=False)

            self.conv_u2b_c = L.Convolution2D(512, 256, ksize=3, stride=1, pad=0, nobias=False)
            self.conv_u2c_d = L.Convolution2D(256, 256, ksize=3, stride=1, pad=0, nobias=False)

            self.upconv_u2d_u1a = L.Deconvolution2D(256, 128, ksize=2, stride=2, pad=0, nobias=False)

            self.conv_u1b_c = L.Convolution2D(256, 128, ksize=3, stride=1, pad=0, nobias=False)
            self.conv_u1c_d = L.Convolution2D(128, 128, ksize=3, stride=1, pad=0, nobias=False)

            # out_channels is 128 due to the original code (not paper)
            self.upconv_u1d_u0a = L.Deconvolution2D(128, 128, ksize=2, stride=2, pad=0, nobias=False)

            self.conv_u0b_c = L.Convolution2D(192, 64, ksize=3, stride=1, pad=0, nobias=False)
            self.conv_u0c_d = L.Convolution2D(64, 64, ksize=3, stride=1, pad=0, nobias=False)

            self.conv_u0d_score = L.Convolution2D(64, 2, ksize=1, stride=1, pad=0, nobias=False)

    def __call__(self, x, t):
        h = cr(self.conv_d0a_b, x)
        assert h.shape[1:] == (64, 570, 570)
        d0c = cr(self.conv_d0b_c, h)
        assert d0c.shape[1:] == (64, 568, 568)

        h = F.max_pooling_2d(d0c, ksize=2, stride=2, pad=0)
        assert h.shape[1:] == (64, 284, 284)

        h = cr(self.conv_d1a_b, h)
        assert h.shape[1:] == (128, 282, 282)
        d1c = cr(self.conv_d1b_c, h)
        assert d1c.shape[1:] == (128, 280, 280)

        h = F.max_pooling_2d(d1c, ksize=2, stride=2, pad=0)
        assert h.shape[1:] == (128, 140, 140)

        h = cr(self.conv_d2a_b, h)
        assert h.shape[1:] == (256, 138, 138)
        d2c = cr(self.conv_d2b_c, h)
        assert d2c.shape[1:] == (256, 136, 136)

        h = F.max_pooling_2d(d2c, ksize=2, stride=2, pad=0)
        assert h.shape[1:] == (256, 68, 68)

        h = cr(self.conv_d3a_b, h)
        assert h.shape[1:] == (512, 66, 66)
        h = cr(self.conv_d3b_c, h)
        assert h.shape[1:] == (512, 64, 64)
        d3c = F.dropout(h, 0.5)

        h = F.max_pooling_2d(d3c, ksize=2, stride=2, pad=0)
        assert h.shape[1:] == (512, 32, 32)

        h = cr(self.conv_d4a_b, h)
        assert h.shape[1:] == (1024, 30, 30)
        h = cr(self.conv_d4b_c, h)
        assert h.shape[1:] == (1024, 28, 28)
        d4c = F.dropout(h, 0.5)

        h = cr(self.upconv_d4c_u3a, d4c)
        assert h.shape[1:] == (512, 56, 56)
        h = F.concat([d3c[:, :, 4:-4, 4:-4], h])
        assert h.shape[1:] == (1024, 56, 56)

        h = cr(self.conv_u3b_c, h)
        assert h.shape[1:] == (512, 54, 54)
        h = cr(self.conv_u3c_d, h)
        assert h.shape[1:] == (512, 52, 52)

        h = cr(self.upconv_u3d_u2a, h)
        assert h.shape[1:] == (256, 104, 104)
        h = F.concat([d2c[:, :, 16:-16, 16:-16], h])
        assert h.shape[1:] == (512, 104, 104)

        h = cr(self.conv_u2b_c, h)
        assert h.shape[1:] == (256, 102, 102)
        h = cr(self.conv_u2c_d, h)
        assert h.shape[1:] == (256, 100, 100)

        h = cr(self.upconv_u2d_u1a, h)
        assert h.shape[1:] == (128, 200, 200)
        h = F.concat([d1c[:, :, 40:-40, 40:-40], h])
        assert h.shape[1:] == (256, 200, 200)

        h = cr(self.conv_u1b_c, h)
        assert h.shape[1:] == (128, 198, 198)
        h = cr(self.conv_u1c_d, h)
        assert h.shape[1:] == (128, 196, 196)

        h = cr(self.upconv_u1d_u0a, h)
        assert h.shape[1:] == (128, 392, 392)
        h = F.concat([d0c[:, :, 88:-88, 88:-88], h])
        assert h.shape[1:] == (192, 392, 392)

        h = cr(self.conv_u0b_c, h)
        assert h.shape[1:] == (64, 390, 390)
        h = cr(self.conv_u0c_d, h)
        assert h.shape[1:] == (64, 388, 388)

        h = self.conv_u0d_score(h)
        assert h.shape[1:] == (2, 388, 388)

        return F.softmax_cross_entropy(h, t)


def cr(c, x):
    return F.relu(c(x))
