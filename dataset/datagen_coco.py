# -*- coding: utf-8 -*-
"""
COCO Dataset train set generator

TODO:
    COCO dataset API will be used to work with this generator

    mpsk	2018-04-02
"""
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import random
import time
import sys
from skimage import transform,io
import scipy.misc as scm
from pycocotools.coco import COCO
import datagen


class DataGenerator(datagen.DataGenerator):
    """ DataGenerator Class :
    Formalized DATA:
        Inputs:
            Inputs have a shape of (Number of Image) X (Height: in_size) X (Width: in_size) X (Channels: 3)
        Outputs:
            Outputs have a shape of (Number of Image) X (Number of Stacks) X (Heigth: self.out_size) X (Width: self.out_size) X (OutputDimendion: 16)
    Joints:
        We use the MPII convention on joints numbering
        List of joints:
            1-'nose'            A
            2-'left_eye'        B
            3-'right_eye'       C
            4-'left_ear'        D
            5-'right_ear'       E
            6-'left_shoulder'   F
            7-'right_shoulder'  G
            8-'left_elbow'      H
            9-'right_elbow'     I
            10-'left_wrist'     J
            11-'right_wrist'    K
            12-'left_hip'       L
            13-'right_hip'      M
            14-'left_knee'      N
            15-'right_knee'     O
            16-'left_ankle'     P
            17-'right_ankle'    Q
            18-'neck'           R
    """

    def __init__(self, joints_list=None, img_dir=None, COCO_anno_file=None, COCO_anno_file_val=None, in_size=368, out_size=None):
        """ Initializer
        Args:
            joints_name			: List of joints condsidered
            img_dir				: Directory containing every images
            train_data_file		: Text file with training set data
            remove_joints		: Joints List to keep (See documentation)
        """
        #   neck will be added in code below
        if joints_list is None:
            #   Neck is added by calculating the skeleton
            self.joints_list = ['nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist', 'l_hip', 'r_hip', 'l_knee','r_knee', 'l_ankle', 'r_ankle', 'neck']
        else:
            self.joints_list = joints_list

        self.in_size = in_size
        if out_size is None:
            self.out_size = self.in_size / 8
        else:
            self.out_size = out_size
        #   we create neck by calculation
        self.joints_num = len(self.joints_list)
        self.letter = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']
        self.img_dir = img_dir
        assert COCO_anno_file is not None
        self.coco_anno = COCO(COCO_anno_file)
        catIds = self.coco_anno.getCatIds(catNms=['person'])
        imgIds = self.coco_anno.getImgIds(catIds=catIds)
        annIds = self.coco_anno.getAnnIds(catIds=catIds)
        self.anno_list = self.coco_anno.loadAnns(annIds)

        self.name = 'coco'

        if COCO_anno_file_val is not None:
            self.coco_anno_val = COCO(COCO_anno_file_val)
            catIds = self.coco_anno_val.getCatIds(catNms=['person'])
            imgIds = self.coco_anno_val.getImgIds(catIds=catIds)
            annIds = self.coco_anno_val.getAnnIds(catIds=catIds)
            self.anno_list_val = self.coco_anno_val.loadAnns(annIds)
        else:
            self.coco_anno_val = None
            self.anno_list_val = None
            print '[!]\tNo Validation Set is Found! The monitoring data might be inaccurate!'
        print '[*]\COCO dataset loaded!'

    # --------------------Generator Initialization Methods ---------------------

    def _reshape_coco(self, joints_flat):
        """ reshape coco annotation in flat list format
            AND Caculate the Neck joint position
        Args:
            joints_flat:    flat format list of joints info directly from COCO
        Return:
            Numpy format matrix of shape of (joint_num * dimension)
        """
        joints = np.zeros((self.joints_num, 2), np.float)
        weight = np.ones((self.joints_num), np.int8)
        r_joints = np.array(joints_flat).reshape((self.joints_num-1,3))[:,:2]
        joints[:-1,:] = r_joints[:,:2]
        joints[np.where(joints==0)] = -1
        #   neck is uncalculatable if any side of shoulder is invisible
        if (joints[5]==-1).any() or (joints[6]==-1).any():
            joints[-1,:] = np.array([-1, -1])
        else:
            joints[-1,:] = (joints[5]+joints[6])/2
        for i in range(joints.shape[0]):
            if(joints[i]==-1).any():
                weight[i] = 0
        return joints, weight

    def _randomize(self):
        """ Randomize the set
        """
        random.shuffle(self.anno_list)

    def generateSet(self, rand=True):
        """ Generate the training and validation set
        Args:
            rand : (bool) True to shuffle the set
        """
        t1 = time.time()
        self.data_dict = []
        self.data_dict_val = []
        if rand:
            self._randomize()
        #   Training data generation
        for m in range(len(self.anno_list)):
            joints, weights = self._reshape_coco(self.anno_list[m]['keypoints'])
            if np.sum(np.logical_and(joints[:,0]!=-1, joints[:,1]!=-1)) < 9:
                # print "[*]\tWarning:\tNo enough joint information in COCO!"
                continue
            _dict = {}
            _dict['weights'] = weights
            _dict['keypoints'] = joints
            _dict['bbox'] = self.anno_list[m]['bbox']
            _dict['image_id'] = self.anno_list[m]['image_id']
            _dict['segmentation'] = self.anno_list[m]['segmentation']
            self.data_dict.append(_dict)
            if m%1000==0:
                sys.stdout.write("\r[>>]\tLoading Annotations... progress: %.2f %%"%(m*100/float(len(self.anno_list))))
                sys.stdout.flush()
        print '\r[*]\t %d Training Annotation loaded! -- %.2f s used.'%(len(self.data_dict), time.time() - t1)
        #   Validation data generation
        t1 = time.time()
        if self.coco_anno_val is not None and self.anno_list_val is not None:
            for m in range(len(self.anno_list_val)):
                joints, weights = self._reshape_coco(self.anno_list_val[m]['keypoints'])
                if np.sum(np.logical_and(joints[:,0]!=-1, joints[:,1]!=-1)) < 9:
                    # print "[*]\tWarning:\tNo enough joint information in COCO!"
                    continue
                _dict = {}
                _dict['weights'] = weights
                _dict['keypoints'] = joints
                _dict['bbox'] = self.anno_list_val[m]['bbox']
                _dict['image_id'] = self.anno_list_val[m]['image_id']
                _dict['segmentation'] = self.anno_list_val[m]['segmentation']
                self.data_dict_val.append(_dict)
                if m%1000==0:
                    sys.stdout.write("\r[>>]\tLoading Annotations... progress: %.2f %%"%(m*100/float(len(self.anno_list_val))))
                    sys.stdout.flush()
            print '\r[*]\t %d Validating Annotation loaded! -- %.2f s used.'%(len(self.data_dict_val), time.time() - t1)


    # ---------------------------- Generating Methods --------------------------

    def _crop_mask(self, mask, padding, crop_box):
        """ Given a bounding box and padding values return cropped image
        Args:
            mask			: Source Image
            padding	: Padding
            crop_box	: Bounding Box
        """
        mask = np.pad(mask, padding[:2], mode='constant')
        max_lenght = max(crop_box[2], crop_box[3])
        mask = mask[int(crop_box[1] - max_lenght//2):int(crop_box[1] + max_lenght//2),
              int(crop_box[0]-max_lenght//2):int(crop_box[0]+max_lenght//2)]
        return mask

    def _augment(self, img, hm, mask, max_rotation=30, angle=None):
        """ # TODO : IMPLEMENT DATA AUGMENTATION
        """
        if random.choice([0, 1]):
            if angle==None:
                r_angle = np.random.randint(-1 * max_rotation, max_rotation)
            else:
                r_angle = angle
            '''
            #   old version
            img = transform.rotate(img, r_angle, preserve_range=True)
            hm = transform.rotate(hm, r_angle)
            rmask = transform.rotate(rmask, r_angle)
            '''
            img = self._rotate_img(img, r_angle)
            hm = self._rotate_hm(hm, r_angle)
            mask = self._rotate_mask(mask, r_angle)
        return img, hm, mask

    # ----------------------- Batch Generator ----------------------------------

    def generate_gt(self, i=0, stacks=4, normalize=True, sample_set='train', ret_dict=None, debug=False):
        """ Auxiliary Generator
        Args:
            See Args section in self._generator
        """
        #   idx     random a index
        #   joints  joints from annotation list
        #   box     bounding box
        #   img, hm, mask
        if sample_set == 'train':
            d_dict = self.data_dict
            d_anno = self.coco_anno
        elif sample_set == 'val':
            d_dict = self.data_dict_val
            d_anno = self.coco_anno_val
        else:
            d_dict = self.data_dict
            d_anno = self.coco_anno
            print "[!]\tNo Set Specificied! Switch to training set instead!"

        idx = random.choice(range(len(d_dict)))
        joints = d_dict[idx]['keypoints']
        while (joints==-1).all():
            print "[!]\tError:\tDetecting null points!"
            idx = random.choice(range(len(d_dict)))
            joints = d_dict[idx]['keypoints']
        box = d_dict[idx]['bbox']
        box[2] = box[0] + box[2]
        box[3] = box[1] + box[3]
        weight = np.asarray(d_dict[idx]['weights'])
        imginfo = d_anno.loadImgs(d_dict[idx]['image_id'])[0]
        img = self.open_img('%s/%s/%s'%(self.img_dir,sample_set+str(2017),imginfo['file_name']))
        mask = d_anno.annToMask(d_dict[idx])
        padd, cbox = self._crop_data(img.shape[0], img.shape[1], box, joints, boxp=0.2)
        new_j = self._relative_joints(cbox, padd, joints, to_size=self.out_size)
        hm = self._generate_hm(self.out_size, self.out_size, new_j, self.out_size, weight)
        img = self._crop_img(img, padd, cbox)
        img = img.astype(np.uint8)
        img = scm.imresize(img, (self.in_size, self.in_size))
        mask = self._crop_mask(mask, padd, cbox)
        mask = scm.imresize(mask, (self.out_size, self.out_size))
        if debug:
            img, hm, mask = self._augment(img, hm, mask, angle=90)
        else:
            img, hm, mask = self._augment(img, hm, mask)
        hm = np.expand_dims(hm, axis=0)
        hm = np.repeat(hm, stacks, axis=0)
        if ret_dict is not None:
                ret_dict[str(i)] = (img, hm, weight, mask)
        i = i + 1
        return img, hm, weight, mask, i

    def _aux_generator(self, batch_size=16, stacks=4, normalize=True, sample_set='train', debug=False):
        """ Auxiliary Generator
        Args:
            See Args section in self.generator

        This generator renders keypoint heatmap, region masks, and training weight(keypoint).
        And the region mask is an instance level attention map which seperate instances by channel

        For details plz see the document inprocceding.
        """
        while True:
            t0 = time.time()
            train_img = np.zeros(
                (batch_size, self.in_size, self.in_size, 3), np.float32)
            train_gtmap = np.zeros((batch_size, stacks, self.out_size, self.out_size, len(
                self.joints_list) + 1), np.float32)
            train_weights = np.zeros(
                (batch_size, len(self.joints_list) + 1), np.float32)
            train_region_mask = np.zeros(
                (batch_size, self.out_size, self.out_size), np.float32)
            i = 0
            while i < batch_size:
                #   generate batch
                img, hm, weight, mask, i = self.generate_gt(i=i,
                                                    stacks=stacks,
                                                    sample_set=sample_set,
                                                    debug=debug)
                i -= 1
                #   feed batch
                if normalize:
                    train_img[i] = img.astype(np.float32) / 255
                else:
                    train_img[i] = img.astype(np.float32)
                train_gtmap[i] = hm
                #   Do we really need such a weight vector ???
                train_weights[i][:len(self.joints_list)] = weight
                train_region_mask[i] = mask
                i += 1

            print "[*]\tGenerate mini-batch of shape %d in %.2f ms" % (batch_size, (time.time()-t0)*1000)
            yield train_img, train_gtmap, train_weights, train_region_mask

    def generator(self, batchSize=16, stacks=4, norm=True, sample_set='train', debug=False):
        """ Create a Sample Generator
        Args:
            batchSize 	: Number of image per batch
            stacks 	 	: Stacks in HG model
            norm 	 	 	: (bool) True to normalize the batch
            sample 	 	: 'train'/'valid' Default: 'train'
        """
        return self._aux_generator(batch_size=batchSize, stacks=stacks, normalize=norm, sample_set=sample_set, debug=debug)

if __name__ == '__main__':
    #   module testing code
    INPUT_SIZE = 368
    IMG_ROOT = "/home/mpsk/data/COCO2017"
    COCO_anno_file = "/home/mpsk/data/COCO2017/annotations/person_keypoints_val2017.json"
    COCO_anno_file_val = "/home/mpsk/data/COCO2017/annotations/person_keypoints_val2017.json"
    gen = DataGenerator(img_dir=IMG_ROOT,
                        COCO_anno_file=COCO_anno_file,
                        COCO_anno_file_val=COCO_anno_file_val,
                        in_size=INPUT_SIZE)
    gen.generateSet(rand=True)
    for n in range(1000):
        img, gtmap, w, mask = next(gen.generator(sample_set='val', debug=True))
        for i in (img, gtmap, w, mask):
            print i.shape
        io.imsave('test.img.jpg', img[0])
        io.imsave("test.joint.jpg", np.meangtmap)
        io.imsave("test.mask.jpg", mask[0])
        print np.average(mask)
        break
