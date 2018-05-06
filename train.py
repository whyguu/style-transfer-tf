import tensorflow as tf
import numpy as np
from skimage import io
from model import *
import argparse
import os


def arg_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, required=False, default='2')
    args = parser.parse_args()

    # config
    log_device_placement = True  # 是否打印设备分配日志
    allow_soft_placement = True  # 如果你指定的设备不存在，允许TF自动分配设备
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.90, allow_growth=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu  # 使用 GPU 0
    config = tf.ConfigProto(log_device_placement=log_device_placement,
                            allow_soft_placement=allow_soft_placement,
                            gpu_options=gpu_options)

    return config


if __name__ == '__main__':
    config = arg_config()
    iter_num = 10001

    model = TransferModel()

    model.build_model()
    loss_op, content_loss_op, style_loss_op = model.calc_loss()

    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(10.0, global_step=global_step, decay_steps=1000, decay_rate=0.6, staircase=False)
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss=loss_op, global_step=global_step, var_list=[model.rlt])
    train_bfgs = tf.contrib.opt.ScipyOptimizerInterface(loss_op, method='L-BFGS-B', options={'maxiter': iter_num})

    with tf.Session(config=config, graph=tf.get_default_graph()) as sess:
        sess.run(tf.global_variables_initializer())
        # 1.
        # train_bfgs.minimize(sess)
        # img = model.get_result_img(sess)
        # io.imsave('./images/output/rlt_bfgs_{:d}.jpg'.format(iter_num), img)
        # exit()
        # 2.
        for i in range(iter_num):
            _, loss, content_loss, style_loss = sess.run(fetches=[train_op, loss_op, content_loss_op, style_loss_op])
            if i % 100 == 0:
                print(i, 'loss: {:.2f}, content loss: {:.2f}, style loss: {:.2f}'.format(loss, content_loss, style_loss))
            if i % 500 == 0:
                print('save a image')
                img = model.get_result_img(sess)
                io.imsave('./images/output/rlt_{:d}.jpg'.format(i), img)




