#!/usr/bin/python
#coding:utf-8

import os
import random
import copy
from multiprocessing import Pool

import cv2
import numpy as np
import tables
import scipy.io as sio
from scipy.stats import multivariate_normal

class Dataset(object):
    def __init__(self, train_list_path=None, img_root=None, anno_root=None, gt_root=None, batch_size=None, in_size=368, joint_num=16, debug=False, pre_gen=False):
        if train_list_path==None or img_root==None or anno_root==None or batch_size==None:
            raise ValueError
        self.train_list = self.__load_list(train_list_path)
        self.img_root = img_root
        self.anno_root = anno_root
        self.gt_root = gt_root
        self.batch_size = batch_size
        self.margin_alpha=0.1
        self.in_size = in_size
        self.debug = debug
        self.idx_batches = None
        self.batch_img = None
        self.batch_gt = None
        self.batch_num = 0
        self.joint_num = joint_num
        self.pre_gen = pre_gen
        '''
        if pre_gen == True:
            self.GenerateAllHeatMaps(parallel=True, procnum=4)
        '''


    def GenerateAllHeatMaps(self, procnum=4):
        p = Pool()
        for name in self.train_list:
            img, gt = self.__gen_pair(name)
            p.apply_async(np.save, 
                args=(self.gt_root + name + ".crop.img.npy", img,))
            p.apply_async(np.save, 
                args=(self.gt_root + name + ".gt.npy", gt,))
            print "[*]\tgenerated hmap of", name
        p.close()
        p.join()
        print "[*]\tdone"
        

    def GenerateOneBatch(self):
        assert self.idx_batches!=None or self.idx_batches!=[]
        current_batch = self.idx_batches[0]
        self.idx_batches.remove(current_batch)
        return self.__struct_mini_batch(current_batch)


    def shuffle(self):
        """
        Shuffle the dataset to batches
        """
        batches = []
        _train_list = copy.deepcopy(self.train_list)
        assert len(_train_list)/self.batch_size>=1
        self.batch_num = len(_train_list)/self.batch_size
        for n in range(len(_train_list)/self.batch_size):
            batch_idx = []
            for m in range(self.batch_size):
                elem = _train_list[random.randint(0, len(_train_list)-1)]
                batch_idx.append(elem)
                _train_list.remove(elem)
            batches.append(batch_idx)
        del _train_list
        self.idx_batches = batches
        if self.debug:
            for n in self.idx_batches:
                print "batch:"
                for m in n:
                    print '\t', m

    def __gen_pair(self, img_name):
        """
        Generate set of GT for one image
        """
        img = cv2.imread(self.img_root + img_name)
        for anno in self.__load_anno(img_name):
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
                h_w[n] = int(bottom_right[n]+self.margin_alpha*img.shape[n]) - int(top_left[n]-self.margin_alpha*img.shape[n])
            a = min([max(h_w), img.shape[0], img.shape[1]])

            if self.debug:
                #   print the crop information
                print "top-left ", top_left, "\tbottom_right ", bottom_right
                print "square of the selected annotations ", a, h_w, max(h_w)
                print "img shape of ", img.shape

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
                    if self.debug:
                        print "position after translation ", anno[n,m] - ([x1,y1])[m]
                    anno[n,m] = anno[n,m] - ([x1,y1])[m]
            GTmaps = np.zeros((self.joint_num+1, self.in_size/8, self.in_size/8), np.float)
            anno_resz = self.__resize_points(anno, img.shape, [self.in_size/8, self.in_size/8])
            #print anno_resz, "\nimage shape is ", img.shape
            GTmap = []
            #   currently choose the first annotation for this image
            for n in range(len(anno_resz)):
                s = int(np.sqrt(self.in_size/8) * self.in_size/8 * 10 / 4096) + 2
                GTmap.append(self.__genGTmap(self.in_size/8, self.in_size/8,
                    anno_resz[n][1], anno_resz[n][0],sigma_h=s, sigma_w=s))
            GTmap.append(np.zeros((self.in_size/8,self.in_size/8),np.float))
            GTmaps += np.array(GTmap)
        if self.debug:
            #   show the croped image
            cv2.imshow("crop", img)
            cv2.waitKey(0)
            #   print anno
            print "anno_origin ", anno
            print "anno_resize ", anno_resz
            #   test gtmap
            _gshow = np.zeros(GTmaps.shape[1:], np.float)
            for n in range(GTmaps.shape[0]):
                _gshow += GTmaps[n]
            print "generate hmap with size ", _gshow.shape
            cv2.imshow("hmap", _gshow)
            cv2.waitKey(0)
        return img, GTmaps.transpose(1,2,0)

    def __struct_mini_batch(self, batch_idx):
        """
        load image and generate GTmap
        Input: 
            1 batch conttain indexs
        """
        batch_img = []
        batch_gt = []
        assert self.idx_batches != None
        for img_name in batch_idx:
            if self.pre_gen == False:
                img, GTmaps = self.__gen_pair(img_name)
            else:
                img = np.load(self.gt_root + img_name + ".crop.img.npy")
                GTmaps = np.load(self.gt_root + img_name + ".gt.npy")
            img_resz = cv2.resize(img, (self.in_size, self.in_size)) / 255.0
            batch_img.append(np.array(img_resz))
            batch_gt.append(GTmaps)
            if self.debug:
                print img.shape
                print np.array(batch_img).shape
                print np.array(batch_gt).shape
        return [batch_img, batch_gt]

    def __test2(dataset_root, img_root, anno_root):
        '''
        #   check the unassigned images
        c = 0
        for n in idx_batches:
            for m in n:
                c += 1
        print len(self.train_list) - c
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

    def __load_list(self, list_path):
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

    def __resize_points(self, points, ori_sz, out_sz):
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

    def __load_anno(self, imgname):
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
        with open(self.anno_root + imgname + '.txt', 'r') as f:
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

    def __test(self, imgname, imgroot, annoroot):
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
#   
            cv2.imshow('test', cv2.resize(img, (640,480)))
            cv2.imshow("GTmap", GTmap)
            cv2.waitKey(0)

    def __genGTmap(self, h, w, pos_x, pos_y, sigma_h=1, sigma_w=1, init=None):
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
    dataset = Dataset(train_list_path="mpii/train_list_test.txt", img_root="mpii/images/", gt_root="mpii/gt/", anno_root="mpii/train/", batch_size=2, debug=True, pre_gen=True)
    dataset.shuffle()
    print "generate batch with size of ", dataset.batch_num
    for n in range(dataset.batch_num):
        dataset.GenerateOneBatch()
