#coding: utf-8
"""
    Convulotional Pose Machine
        For Single Person Pose Estimation
    Human Pose Estimation Project in Lab of IP
    Author: Liu Fangrui aka mpsk
        Beijing University of Technology
            College of Computer Science & Technology
    Experimental Code
        !!DO NOT USE IT AS DEPLOYMENT!!

    @mpsk:  Feel free to use except you have commercial purpose
"""
import cv2
import numpy as np
import tensorflow as tf

import sys
sys.path.append("..")
from net.Layers import LayerLibrary


sess = tf.Session()
layer = LayerLibrary()
layer.in_size = 368
img_in = tf.placeholder(tf.float32, shape=[None, 46, 46, 3])
bbox_in = tf.placeholder(tf.float32, shape=[None, 4])
ind_in = tf.placeholder(tf.int32, shape=[None])

crop = layer.roi_align(img_in, bbox_in, ind_in, 46, 3)
print crop.shape
dispa, _ = layer.dispatch_layer(bbox_in, crop, ind_in, 2, 3, 46, 3)
print dispa.shape
patch = layer.patch_with_crop_and_resize(crop[0], bbox_in[0], 46, 368)

def test(test_num=10):

    img_list = []
    name_list = ['../test.1.png']
    for name in name_list:
        img = cv2.imread(name)
        img_list.append(cv2.resize(img, (46, 46)))
        print "[=]\t", img.shape
    img_batch = np.stack(img_list, axis=0)
    #   cx, cy, w, h
    #bbox_batch = np.array([[61, 55, 70, 75], [0, 0, 30, 50], [0, 47, 40, 20]], dtype=np.float32)
    bbox_batch = np.random.randint(-100, high=100, size=(img_batch.shape[0]*test_num, 4))
    ind_batch = np.repeat(range(0, img_batch.shape[0]), test_num)
    print bbox_batch
    print ind_batch.shape

    dispa_out = sess.run(dispa, feed_dict={img_in:img_batch, 
                                            bbox_in:bbox_batch,
                                            ind_in: ind_batch})
    #   Post processing
    for n in range(len(img_list)):
        cv2.imwrite('roi_dispatch.ori.jpg', img_list[0])
    for n in range(img_batch.shape[0]):
        cv2.imwrite('roi_dispatch.batch.'+ str(ind_batch[n]) + '.' + str(n) + '.jpg', np.max(dispa_out[n, :, :, :], axis=-1))
        print dispa_out.shape


if __name__ == '__main__':
    for n in range(10):
        test(test_num=5)
