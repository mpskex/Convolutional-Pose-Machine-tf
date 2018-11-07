# coding: utf-8
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

import time

import sys
sys.path.append("..")
import net.PPNnMask as PPNnMask

from realtime_demo import RealTimeDemo


def build_anchors(anchor_base_size, anchor_ratio, anchor_scale):
    anchors = []
    for scale in anchor_scale:
        for ratio in anchor_ratio:
            if ratio != 1:
                for r in [float(ratio), 1/float(ratio)]:
                    size = anchor_base_size * anchor_base_size
                    size_ratio = size / r
                    ws = np.round(np.sqrt(size_ratio))
                    hs = np.round(ws * r)
                    anchors.append((int(scale * ws), int(scale * hs)))
            else:
                size = anchor_base_size * anchor_base_size
                size_ratio = size / ratio
                ws = np.round(np.sqrt(size_ratio))
                hs = np.round(ws * ratio)
                anchors.append((int(scale * ws), int(scale * hs)))
    return anchors

class PPN_RTD(RealTimeDemo):
    """ PPN Testing Module
    """
    def __draw_bbox__(self, img, bboxes, color=(255,0,0)):
        """ Draw Bounding box
        Args:
            img     :   input image
            bboxes  :   list of bounding boxes for this image
        Return:
            img     :   image with bbox visualization   
        """
        for bbox in bboxes:
            bbox = np.array(bbox).astype(np.int16)
            print bbox
            cv2.rectangle(img, (int(bbox[1] - bbox[2]//2), (int(bbox[0] - bbox[3]//2))),
                            (int(bbox[1] + bbox[2]//2), int(bbox[0] + bbox[3]//2)), color)
            '''
            cv2.rectangle(img, (int(bbox[0] - bbox[2]//2), (int(bbox[1] - bbox[3]//2))),
                            (int(bbox[0] + bbox[2]//2), int(bbox[1] + bbox[3]//2)), color)
            '''
        return img
    def __estimate__(self, img):
        """ Estimate a img's result through a given model
            For this case we estimate PPN's Accuracy 
        Note:   This function should be reload to fit other model's output

        Args:
            img         :   given_img
        Return:
            ret_img     :   visualized result
            t_pred      :   Prediction time stamp for calculating time cost
            t_postp     :   Post Processing time stamp for calculating time cost
        """
        #   Note that
        img = cv2.resize(img, (self.model.in_size, self.model.in_size))
        bbox = self.model.sess.run(self.model.ppn_result, feed_dict={self.model.img: np.expand_dims(img, axis=0)/255.0})
        '''
        offset = self.model.sess.run(self.model.ppn_bbox_pred, feed_dict={self.model.img: np.expand_dims(img, axis=0)})[0]
        offset = np.reshape(offset, (self.model.out_size, self.model.out_size, len(self.model.anchors), 4))
        print offset[:, :, 0, :]
        '''
        bbox = bbox[np.where(bbox[:,-1]==0)]
        t_pred = time.time()
        # img = cv2.resize(img, (self.model.out_size, self.model.out_size))
        img = self.__draw_bbox__(img, ((self.model.in_size/float(self.model.out_size)) * bbox).tolist())
        t_postp = time.time()
        return img, t_pred, t_postp

if __name__ == '__main__':

    #'''
    model = PPNnMask.PPNnMask(pretrained_model='../model/PPNv8@mpii_lr0.0004_insize368/model.npy',
                    stage=1,
                    load_pretrained=True,
                    training=False,
                    rois_max=16,
                    anchors=build_anchors(2,
                                [1, 2],
                                [1, 4, 8]),
                    name='PPNv8@mpii',
                    _type='V8'
                )
    #'''
    model.BuildModel(no_mask_net=True)
    ppn_demo = PPN_RTD(model)
    ppn_demo.run()
