from layers import *

img_height = 256  # 图像高度
img_width = 256  # 图像宽度
img_layer = 3  # 图像通道
img_size = img_height * img_width  # 图像尺寸

batch_size = 1  # 一个批次的数据中图像的个数

ngf = 32
ndf = 64


def build_resnet_block(inputres, dim, name="resnet"):
    with tf.variable_scope(name):
        out_res = tf.pad(inputres, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        _, out_res = general_conv2d(out_res, dim, 3, 3, 1, 1, 0.02, "VALID", "c1", relufactor=0.2)
        out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        _, out_res = general_conv2d(out_res, dim, 3, 3, 1, 1, 0.02, "VALID", "c2", do_relu=False)

        return lrelu(out_res + inputres)


def build_generator_resnet_9blocks(inputgen, name="generator"):
    with tf.variable_scope(name):
        f = 7
        ks = 3

        # noise = tf.get_variable(name="noise", dtype="tf.float32", trainable=False, shape=[1, 256, 256, 3],
        #                         initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0))
        # input = tf.add(inputgen, noise)

        pad_input = tf.pad(inputgen, [[0, 0], [ks, ks], [ks, ks], [0, 0]], "REFLECT")
        norm1, o_c1 = general_conv2d(pad_input, ngf, f, f, 1, 1, 0.02, name="c1", relufactor=0.2)
        norm2, o_c2 = general_conv2d(o_c1, ngf * 2, ks, ks, 2, 2, 0.02, "SAME", "c2", relufactor=0.2)
        norm3, o_c3 = general_conv2d(o_c2, ngf * 4, ks, ks, 2, 2, 0.02, "SAME", "c3", relufactor=0.2)

        o_r1 = build_resnet_block(o_c3, ngf * 4, "r1")
        o_r2 = build_resnet_block(o_r1, ngf * 4, "r2")
        o_r3 = build_resnet_block(o_r2, ngf * 4, "r3")
        o_r4 = build_resnet_block(o_r3, ngf * 4, "r4")
        o_r5 = build_resnet_block(o_r4, ngf * 4, "r5")
        o_r6 = build_resnet_block(o_r5, ngf * 4, "r6")
        o_r7 = build_resnet_block(o_r6, ngf * 4, "r7")
        o_r8 = build_resnet_block(o_r7, ngf * 4, "r8")
        o_r9 = build_resnet_block(o_r8, ngf * 4, "r9")

        norm4, _ = general_deconv2d(o_r9, [batch_size, 128, 128, ngf * 2], ngf * 2, ks, ks, 2, 2, 0.02, "SAME", "c4")
        o_c4_c2 = tf.concat(axis=3, values=[norm4, norm2], name="o_c4_c2")
        _, o_c4 = general_conv2d(o_c4_c2, ngf * 2, ks, ks, 1, 1, 0.02, "SAME", "o_c4_merge")
        norm5, _ = general_deconv2d(o_c4, [batch_size, 256, 256, ngf], ngf, ks, ks, 2, 2, 0.02, "SAME", "c5")
        o_c5_c1 = tf.concat(axis=3, values=[norm5, norm1], name="o_c5_c1")
        _, o_c5 = general_conv2d(o_c5_c1, ngf, ks, ks, 1, 1, 0.02, "SAME", "o_c5_merge")
        norm6, _ = general_conv2d(o_c5, img_layer, f, f, 1, 1, 0.02, "SAME", "c6")
        o_c6_input = tf.concat(axis=3, values=[norm6, inputgen], name="o_c6_input")
        _, o_c6 = general_conv2d(o_c6_input, img_layer, f, f, 1, 1, 0.02, "SAME", "o_c6_merge", do_relu=False)

        out_gen = tf.nn.tanh(o_c6, "t1")

        return out_gen


def build_gen_discriminator(inputdisc, name="discriminator"):
    with tf.variable_scope(name):
        f = 4

        _, o_c1 = general_conv2d(inputdisc, ndf, f, f, 2, 2, 0.02, "SAME", "c1", do_norm=False, relufactor=0.2)
        _, o_c2 = general_conv2d(o_c1, ndf * 2, f, f, 2, 2, 0.02, "SAME", "c2", relufactor=0.2)
        _, o_c3 = general_conv2d(o_c2, ndf * 4, f, f, 2, 2, 0.02, "SAME", "c3", relufactor=0.2)
        _, o_c4 = general_conv2d(o_c3, ndf * 8, f, f, 1, 1, 0.02, "SAME", "c4", relufactor=0.2)
        _, o_c5 = general_conv2d(o_c4, 1, f, f, 1, 1, 0.02, "SAME", "c5", do_norm=False, do_relu=False)

        return o_c5
