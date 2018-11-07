# -*- coding: utf-8 -*-
"""
Deep Human Pose Estimation

Project by Walid Benbihi
MSc Individual Project
Imperial College
Created on Wed Jul 12 15:53:44 2017

@author: Walid Benbihi
@mail : w.benbihi(at)gmail.com
@github : https://github.com/wbenbihi/hourglasstensorlfow/

Abstract:
        This python code creates a Stacked Hourglass Model
        (Credits : A.Newell et al.)
        (Paper : https://arxiv.org/abs/1603.06937)

        Code translated from 'anewell' github
        Torch7(LUA) --> TensorFlow(PYTHON)
        (Code : https://github.com/anewell/pose-hg-train)

        Modification are made and explained in the report
        Goal : Achieve Real Time detection (Webcam)
        ----- Modifications made to obtain faster results (trade off speed/accuracy)

        This work is free of use, please cite the author if you use it!
========================================================================
P.S.:
    This is a modified version of the origin HG model
    It is free to scale up and down and more flexible for experiments
    Net Structure might be different from the master branch

NOTE:
    Embedding data generator for the whole architecture
    Generate keypoint heatmap, region mask, weight

    mpsk 09-08 2018
"""
import numpy as np
import cv2
import random
import math
import time
import scipy.misc as scm
from skimage.draw import line_aa
from skimage import transform, io

from functools import partial
from multiprocessing import Pool, Process, Manager

import datagen_coco


def unwrap_self_gen_gt(arg, **kwarg):
    return arg[0].generate_gt(*arg[1:], **kwarg)

class DataGenerator(datagen_coco.DataGenerator):
    """ Embedding Data Generator

        This Generator is designed to ground truth to new structure
    """
    def __init__(self, *args, **kwargs):
        super(DataGenerator, self).__init__(*args, **kwargs)

    def _aux_generator(self, batch_size=16, stacks=4, normalize=True, cpu_count=8, debug=False, sample_set='train'):
        """ Auxiliary Generator
        Args:
            See Args section in self.generator

        This generator renders keypoint heatmap, region masks, and training weight(keypoint).
        And the region mask is an instance level attention map which seperate instances by channel

        For details plz see the document inprocceding.

        """
        while True:
            t0 = time.time()
            train_img = []
            train_gtmap = []
            train_weights = []
            train_region_mask = []

            # DOING: add multiprocessing
            # add parameters for function run
            # return value and update total yield value
            jobs = []
            manager = Manager()
            ret_dict = manager.dict()
            for i in range(batch_size):
                p = Process(target=self.generate_gt, args=(i, stacks, True, sample_set, ret_dict, debug))
                jobs.append(p)
            c = 0
            while c < batch_size:
                for proc_num in range(2*cpu_count):
                    if c+proc_num < len(jobs):
                        job_count = proc_num
                        jobs[c+proc_num].start()
                for proc_num in range(2*cpu_count):
                    if c+proc_num < len(jobs):
                        jobs[c+proc_num].join()
                c += job_count + 1

            for rec in range(batch_size):
                img, hm, weight, rmask = ret_dict[str(rec)]
                #   feed batch
                if normalize:
                    train_img.append(img.astype(np.float32) / 255)
                else:
                    train_img.append(img.astype(np.float32))
                train_gtmap.append(hm)
                #   Do we really need such a weight vector ???
                train_weights.append(weight)
                train_region_mask.append(rmask)
            
            assert len(train_img) == len(train_gtmap) == len(train_weights) ==len(train_region_mask)

            print "[*]\tGenerate mini-batch of shape %d in %.2f ms" % (
                batch_size, (time.time()-t0)*1000)
            #   submit all batch gt
            #   train_bbox_offset -> box_nms
            #   train_iou_map -> ppn_cls_score
            yield np.array(train_img), \
                    np.array(train_gtmap), \
                    np.array(train_weights), \
                    np.array(train_region_mask)


if __name__ == '__main__':
    #   module testing code
    INPUT_SIZE = 368
    IMG_ROOT = "/home/mpsk/data/COCO2017"
    # COCO_anno_file = "/home/mpsk/data/COCO2017/annotations/person_keypoints_train2017.json"
    COCO_anno_file = "/home/mpsk/data/COCO2017/annotations/person_keypoints_val2017.json"
    COCO_anno_file_val = "/home/mpsk/data/COCO2017/annotations/person_keypoints_val2017.json"
    gen = DataGenerator(img_dir=IMG_ROOT,
                        COCO_anno_file=COCO_anno_file,
                        COCO_anno_file_val=COCO_anno_file_val,
                        in_size=INPUT_SIZE,
                        )
    gen.generateSet(rand=True)
    img, gtmap, _, rmask = next(gen._aux_generator(batch_size=12, normalize=True, debug=True, sample_set='val'))
    print "[*]\tMask has shape of ", rmask.shape
    for index in range(img.shape[0]):
        rsz_img = cv2.resize(img[index], (gen.out_size, gen.out_size))
        io.imsave('gt' + str(index) + '.jpg', np.sum(
            gtmap[index, 0, :, :, :-1]*100, axis=-1).astype(np.uint8))
        print "[>>]\tsaved image"
