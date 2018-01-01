import tensorflow as tf

from layers import leaky_relu
from layers import conv2d
from layers import deconv2d

img_height = 256  # 图像高度
img_width = 256  # 图像宽度
img_layer = 3  # 图像通道
img_size = img_height * img_width  # 图像尺寸

batch_size = 1  # 一个批次的数据中图像的个数

ngf = 32
ndf = 64

LAMBDA = 1.0  # 从 encoder 向 decoder 传递数据时的权重


def resnet_block(inputres, dim, name="resnet"):
    with tf.variable_scope(name):
        out_res = tf.pad(inputres, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        _, out_res = conv2d(out_res, dim, 3, 3, 1, 1, 0.02, "VALID", "c1", relufactor=0.2)
        out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        _, out_res = conv2d(out_res, dim, 3, 3, 1, 1, 0.02, "VALID", "c2", do_relu=False)

        return leaky_relu(out_res + inputres)


def generator(inputgen, name="generator"):
    with tf.variable_scope(name):
        f = 7
        ks = 3

        pad_input = tf.pad(inputgen, [[0, 0], [ks, ks], [ks, ks], [0, 0]], "REFLECT")
        norm1, o_c1 = conv2d(pad_input, ngf, f, f, 1, 1, 0.02, name="encoder1", relufactor=0.2)
        norm2, o_c2 = conv2d(o_c1, ngf * 2, ks, ks, 2, 2, 0.02, "SAME", "encoder2", relufactor=0.2)
        norm3, o_c3 = conv2d(o_c2, ngf * 4, ks, ks, 2, 2, 0.02, "SAME", "encoder3", relufactor=0.2)

        o_r1 = resnet_block(o_c3, ngf * 4, "r1")
        o_r2 = resnet_block(o_r1, ngf * 4, "r2")
        o_r3 = resnet_block(o_r2, ngf * 4, "r3")
        o_r4 = resnet_block(o_r3, ngf * 4, "r4")
        o_r5 = resnet_block(o_r4, ngf * 4, "r5")
        o_r6 = resnet_block(o_r5, ngf * 4, "r6")

        norm4, _ = deconv2d(o_r6, ngf * 2, ks, ks, 2, 2, 0.02, "SAME", "decoder1")
        o_c4_c2 = tf.concat(axis=3, values=[norm4, norm2])
        _, o_c4 = conv2d(o_c4_c2, ngf * 2, ks, ks, 1, 1, 0.02, "SAME", "o_c4_merge")

        norm5, _ = deconv2d(o_c4, ngf, ks, ks, 2, 2, 0.02, "SAME", "decoder2")
        o_c5_c1 = tf.concat(axis=3, values=[norm5, norm1])
        _, o_c5 = conv2d(o_c5_c1, ngf, ks, ks, 1, 1, 0.02, "SAME", "o_c5_merge")

        norm6, _ = conv2d(o_c5, img_layer, f, f, 1, 1, 0.02, "SAME", "output")
        o_c6_input = tf.concat(axis=3, values=[norm6, inputgen])
        _, o_c6 = conv2d(o_c6_input, img_layer, f, f, 1, 1, 0.02, "SAME", "o_c6_merge", do_relu=False)

        out_gen = tf.nn.tanh(x=o_c6)

        return out_gen


def discriminator(inputdisc, name="discriminator"):
    with tf.variable_scope(name):
        f = 3

        patch_input = tf.random_crop(inputdisc, [1, 70, 70, 3])
        _, o_c1 = conv2d(patch_input, ndf, f, f, 2, 2, 0.02, "SAME", "disc1", do_norm=False, relufactor=0.2)
        _, o_c2 = conv2d(o_c1, ndf * 2, f, f, 2, 2, 0.02, "SAME", "disc2", relufactor=0.2)
        _, o_c3 = conv2d(o_c2, ndf * 4, f, f, 2, 2, 0.02, "SAME", "disc3", relufactor=0.2)
        _, o_c4 = conv2d(o_c3, ndf * 8, f, f, 1, 1, 0.02, "SAME", "disc4", relufactor=0.2)
        _, o_c5 = conv2d(o_c4, 1, f, f, 1, 1, 0.02, "SAME", "disc5", do_norm=False, do_relu=False)

        return o_c5
