import threading

import tensorflow as tf
import os
import numpy as np
from datetime import datetime
import time


model_path = os.path.join("../model", 'classify_image_graph_def.pb')


def create_default_graph():
    with tf.io.gfile.GFile(model_path, 'rb') as f:
        for i in range(5):
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')


def get_list_of_image_data(path):
    files = os.listdir(path)
    images_data = []
    for f in files:
        file_name = '%s/%s' % (path, os.path.basename(f))
        print(file_name)
        if not os.path.isfile(file_name) or not f.endswith(".jpg"):
            continue

        img_data = tf.gfile.FastGFile(file_name, 'rb').read()
        images_data.append(img_data)
    return images_data


def process_multi_imgs(images_path):
    images_data = get_list_of_image_data(images_path)
    print('========images_data LEN:  %d ' % len(images_data))

    beg_time = datetime.now()
    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')  # 从计算图中提取张量
        print(softmax_tensor)
        for i in range(len(images_data)):
            time1 = datetime.now()
            image_data = images_data[i]
            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
            time2 = datetime.now()
            print('softmax_tensor计算时间为：%s' % (time2 - time2))
            predictions = np.squeeze(predictions[0])  # 去掉冗余的1维形状，比如把张量形状从(1,3,1)变为(3)
            top5 = predictions.argsort()[-5:][::-1]
            time3 = datetime.now()
            print('图片预测计算时间为：%s' % (time3 - time1))

    end_time = datetime.now()
    print('Total_Time for one session: %s ' % (end_time - beg_time))


if __name__ == '__main__':
    print(tf.__version__)
    print(tf.__path__)

    create_default_graph()

    process_multi_imgs('../images/10')
