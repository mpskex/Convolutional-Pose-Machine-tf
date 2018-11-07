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

def test(bbox=None, test_num=100):
    sess = tf.Session()
    #sess = tf_debug.TensorBoardDebugWrapperSession(sess, "iLab-K40:6064")
    layer = LayerLibrary()
    layer.in_size = 64

    img_in = tf.placeholder(tf.float32, shape=[None, 46, 46, 3])
    bbox_in = tf.placeholder(tf.float32, shape=[None, 4])
    ind_in = tf.placeholder(tf.int32, shape=[None])
    
    crop = layer.roi_align(img_in, bbox_in, ind_in, 46)
    patch = layer.patch_with_crop_and_resize(crop[0], bbox_in[0], 46, layer.in_size)

    img_list = []
    name_list = ['../test.1.png']
    for name in name_list:
        img = cv2.imread(name)
        img_list.append(cv2.resize(img, (46, 46)))
        print "[=]\t", img.shape
    img_batch = np.stack(img_list, axis=0)
    #   cx, cy, w, h
    ind_batch = np.array(range(img_batch.shape[0]))

    cv2.imwrite('roi_dispatch.ori.jpg', img_batch[0])
    if bbox is None:
        for n in range(test_num):
            bbox_batch = np.random.randint(-100, high=100, size=(img_batch.shape[0], 4))
            print "[<<]", str(n), "-th ", bbox_batch
            patch_out= sess.run(patch, feed_dict={img_in:img_batch, 
                                                                    bbox_in:bbox_batch,
                                                                    ind_in: ind_batch})
            #   Post processing
            print "[>>]\t  patch_out.shape:\t", patch_out.shape
            #cv2.imwrite('roi_dispatch.crop.'+str(n)+'.jpg', crop_out[0])
            cv2.imwrite('roi_dispatch.patch.'+str(n)+'.jpg', patch_out)
    elif len(bbox) == 4:
        n = 0
        bbox_batch = np.array([bbox], np.float)
        print "[<<]", str(n), "-th ", bbox_batch
        patch_out= sess.run(patch, feed_dict={img_in:img_batch, 
                                                                bbox_in:bbox_batch,
                                                                ind_in: ind_batch})
        #   Post processing
        print "[>>]\t  patch_out.shape:\t", patch_out.shape
        #cv2.imwrite('roi_dispatch.crop.'+str(n)+'.jpg', crop_out[0])
        cv2.imwrite('roi_dispatch.patch.'+str(n)+'.jpg', patch_out)

if __name__ == '__main__':
    #test([38,0,-47,96])
    test(test_num=1000)