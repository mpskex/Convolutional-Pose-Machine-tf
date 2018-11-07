#!/usr/bin/python
#coding:utf-8

import os
import random
import cv2
import numpy as np
import tables
import scipy.io as sio
from scipy.stats import multivariate_normal

from Global import *

"""
    *	load_list(list_path)
    *	resize_points(points, ori_sz, out_sz)
    *	load_anno(img_name, imgroot, annoroot)
    *	genGTmap(h, w, pos_x, pos_y, ...)

    mpsk@github
"""


def shuffle(train_list, batch_size):
    """
    Shuffle the dataset to batches
    Input:
        train_list: whole training list
        batch_size: default to 64
    Ouput:
        batches: small batches contain indexs
    """
    batches = []
    for n in range(len(train_list)/batch_size):
        batch_idx = []
        for m in range(batch_size):
            elem = train_list[random.randint(0, len(train_list)-1)]
            batch_idx.append(elem)
            train_list.remove(elem)
        batches.append(batch_idx)
    return batches


def __struct_mini_batch(idx_batch, margin_alpha=0.1, debug=False):
    """
    load image and generate GTmap
    Input: 
        1 batch conttain indexs
    Output:
        1 batch contain image and GTmap
    """
    batch_img = []
    batch_gt = []
    for img_name in idx_batch:
        img = cv2.imread(IMG_ROOT + img_name)
        for anno in load_anno(img_name, IMG_ROOT, ANNO_ROOT):
            assert anno.shape == (16, 2)
            top_left = [img.shape[0], img.shape[1]]
            bottom_right = [0, 0]
            for n in range(anno.shape[0]):
                for m in range(anno.shape[1]):
                    if anno[n,m] > 0:
                        if anno[n,m] > bottom_right[m]:
                            bottom_right[m] = anno[n,m]
                        if anno[n,m] < top_left[m]:
                            top_left[m] = anno[n,m]
            assert len(top_left) == len(bottom_right) == 2
            
            h_w = [0, 0]
            for n in range(len(top_left)):
                h_w[n] = int(bottom_right[n]+margin_alpha*img.shape[n]) - int(top_left[n]-margin_alpha*img.shape[n])
            a = min([max(h_w), img.shape[0], img.shape[1]])

            if debug:
                #   print the crop information
                print top_left, "\t", bottom_right
                print a, h_w, max(h_w)
                print img.shape

            y1 = int((bottom_right[1]+top_left[1])/2-a/2)
            y2 = int((bottom_right[1]+top_left[1])/2+a/2)
            x1 = int((bottom_right[0]+top_left[0])/2-a/2)
            x2 = int((bottom_right[0]+top_left[0])/2+a/2)

            def __f(x):
                if x<=0:
                    x = 0
                return x
            x1 = __f(x1)
            y1 = __f(y1)
            x2 = __f(x2)
            y2 = __f(y2)
            
            assert x1 >= 0 and y1 >= 0
            #   crop
            img = img[y1:y2, x1:x2, :]
            #   add to batch_img

            assert min(img.shape) > 0

            for n in range(anno.shape[0]):
                for m in range(anno.shape[1]):
                    #assert anno[n,m] > (bottom_right[1]+top_left[1])/2-a/2 
                    if debug:
                        print anno[n,m] - ([x1,y1])[m]
                    anno[n,m] = anno[n,m] - ([x1,y1])[m]
            GTmaps = np.zeros((16, INPUT_SIZE/8, INPUT_SIZE/8), np.float)

            anno_resz = resize_points(anno, img.shape, [INPUT_SIZE/8, INPUT_SIZE/8])
            #print anno_resz, "\nimage shape is ", img.shape
            GTmap = []
            #   currently choose the first annotation for this image
            for n in range(len(anno_resz)):
                GTmap.append(genGTmap(INPUT_SIZE/8, INPUT_SIZE/8, anno_resz[n][1], anno_resz[n][0]))
            GTmaps += np.array(GTmap)
        if debug:
            #   show the croped image
            cv2.imshow("crop", img)
            cv2.waitKey(0)
            #   print anno
            print anno
            print anno_resz
            #   test gtmap
            _gshow = np.zeros(GTmaps.shape[1:], np.float)
            for n in range(GTmaps.shape[0]):
                _gshow += GTmaps[n]
            print "generate hmap with size ", _gshow.shape
            cv2.imshow("hmap", _gshow)
            cv2.waitKey(0)
        GTmaps = GTmaps.transpose(1,2,0)
        img_resz = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
        batch_img.append(np.array(img_resz))
        batch_gt.append(GTmaps)
        if debug:
            print img.shape
            print np.array(batch_img).shape
            print np.array(batch_gt).shape
    return [batch_img, batch_gt]

def test2(dataset_root, img_root, anno_root):
    '''
    #   check the unassigned images
    c = 0
    for n in idx_batches:
        for m in n:
            c += 1
    print len(train_list) - c
    '''
    '''
    #   see the if there is a duplicated set of batch
    flag = 0
    for n in idx_batches:
        for m in idx_batches:
            for k in n:
                if k in m and m is not n:
                    flag = 1
    print "flag is ", flag
    '''



def load_list(list_path):
    """
    function to load the annotation list from the mpii dataset
    load as python list which support the random pick
    """
    li = []
    with open(list_path, 'r') as f:
        li = f.readlines()
        f.close()
    for n in range(len(li)):
        li[n] = li[n].split('\r\n')[0]
    return li

def resize_points(points, ori_sz, out_sz):
    """
    function to transfer the points from origin image

    Input:
        ponits: k * 2(dimension of point) numpy array
        ori_sz:	2 numpy array / python list
            **WATCHOUT** Here shape formation is [h, w]
        out_sz: 2 numpy array / python list
            formation is not that important
            cuz here we use the image as square
    Ouput:
        points: k * 2(dimension of point) numpy array
    """
    assert points.shape == (16,2)
    ret = np.zeros(points.shape, np.float)
    for m in range(points.shape[0]):
        ret[m][0] = (points[m][0]/ori_sz[1]) * out_sz[1]
        ret[m][1] = (points[m][1]/ori_sz[0]) * out_sz[0]
    return ret

def load_anno(imgname, imgroot, annoroot):
    """
    Function to load trainable data set 
    annotation is to proceed with numpy structure
    Also there is a annotation rect to output
    to keep the image width-height ratio unchanged
    this would help the net to learn from struct...
        maybe

    Ouput:
        ret
    """
    null_flag = 0
    li = []
    ret = []
    with open(annoroot + imgname + '.txt', 'r') as f:
        li = f.readlines()
        f.close()
    for n in range(len(li)):
        points = np.zeros((16,2), np.float)
        elem = li[n].split('\t')
        for t in range((len(elem)-1)/2):
            if elem[2*t] != 'nan' and elem[2*t+1] != 'nan':
                points[t][0] = round(float(elem[2*t]))
                points[t][1] = round(float(elem[2*t+1]))
            else:
                #print "[*]\tnull point detected"
                null_flag = 1
                points[t][0] = -1
                points[t][1] = -1
        ret.append(points)
        #print "\tsize of ", len(ret[0])
    return ret

def test(imgname, imgroot, annoroot):
    """
    Function to test the effectivity of those functions
    it would load and draw the annotation on the image
    """
    img = cv2.imread(imgroot + imgname)
    GTmap = np.zeros((46,46), np.float)
    if not img.any:
        print "Error Load Image!"
    else:
        anno = load_anno(imgname, imgroot, annoroot)
        for t in range(len(anno)):
            print  type(anno[t])
            t_anno = resize_points(anno[t], img.shape, [46, 46])
            for n in range(len(anno[t])):
                print img.shape
                GTmap += genGTmap(46, 46, t_anno[n][1], t_anno[n][0])
                cv2.circle(img, (int(anno[t][n][0]), int(anno[t][n][1])), 5, (0, 0, 255), 2)
                print GTmap
        cv2.imshow('test', cv2.resize(img, (640,480)))
        cv2.imshow("GTmap", GTmap)
        cv2.waitKey(0)

def genGTmap(h, w, pos_x, pos_y, sigma_h=1, sigma_w=1, init=None):
    """
    Compute the heat-map of size (w x h) with a gaussian distribution fit in
    position (pos_x, pos_y) and a convariance matix defined by the related
    sigma values.
    The resulting heat-map can be summed to a given heat-map init.
    """
    if pos_x>0 and pos_y>0:
        init = init if init is not None else []

        cov_matrix = np.eye(2) * ([sigma_h**2, sigma_w**2])

        x, y = np.mgrid[0:h, 0:w]
        pos = np.dstack((x, y))
        rv = multivariate_normal([pos_x, pos_y], cov_matrix)

        tmp = rv.pdf(pos)
        hmap = np.multiply(
            tmp, np.sqrt(np.power(2 * np.pi, 2) * np.linalg.det(cov_matrix))
        )
        idx = np.where(hmap.flatten() <= np.exp(-4.6052))
        hmap.flatten()[idx] = 0

        if np.size(init) == 0:
            return hmap

        assert (np.shape(init) == hmap.shape)
        hmap += init
        idx = np.where(hmap.flatten() > 1)
        hmap.flatten()[idx] = 1
        return hmap
    else:
        return np.zeros((h, w))

if __name__ == '__main__':
    batch_size = 10

    train_list =load_list(MPII_ROOT + "train_list.txt")
    idx_batches = shuffle(train_list[:], batch_size)
    _train_batch = __struct_mini_batch(idx_batches[0], debug=True)
