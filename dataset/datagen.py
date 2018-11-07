# -*- coding: utf-8 -*-
"""
TODO:
    Data generater should be able to generate bounding box
    to compensate the error during multi-persion sample training
    This would help net get normalized and balanced very quickly

    This Arch can be use in Densepose complementation(need to be tested)

    mpsk	2018-03-02
"""
import numpy as np
import cv2
import os
import math
import matplotlib.pyplot as plt
import random
import time
from skimage import transform
import scipy.misc as scm
import skimage.io as io
import skimage.measure as msr
from numba import jit

class DataGenerator(object):
    """ DataGenerator Class : To generate Train, Validatidation and Test sets
    for the Deep Human Pose Estimation Model 
    Formalized DATA:
        Inputs:
            Inputs have a shape of (Number of Image) X (Height: 256) X (Width: 256) X (Channels: 3)
        Outputs:
            Outputs have a shape of (Number of Image) X (Number of Stacks) X (Heigth: self.out_size) X (Width: self.out_size) X (OutputDimendion: 16)
    Joints:
        We use the MPII convention on joints numbering
        List of joints:
            00 - Right Ankle
            01 - Right Knee
            02 - Right Hip
            03 - Left Hip
            04 - Left Knee
            05 - Left Ankle
            06 - Pelvis (Not present in other dataset ex : LSP)
            07 - Thorax (Not present in other dataset ex : LSP)
            08 - Neck
            09 - Top Head
            10 - Right Wrist
            11 - Right Elbow
            12 - Right Shoulder
            13 - Left Shoulder
            14 - Left Elbow
            15 - Left Wrist
    # TODO : Modify selection of joints for Training
    
    How to generate Dataset:
        Create a TEXT file with the following structure:
            image_name.jpg[LETTER] box_xmin box_ymin box_xmax b_ymax joints
            [LETTER]:
                One image can contain multiple person. To use the same image
                finish the image with a CAPITAL letter [A,B,C...] for 
                first/second/third... person in the image
             joints : 
                Sequence of x_p y_p (p being the p-joint)
                /!\ In case of missing values use -1
                
    The Generator will read the TEXT file to create a dictionnary
    Then 2 options are available for training:
        Store image/heatmap arrays (numpy file stored in a folder: need disk space but faster reading)
        Generate image/heatmap arrays when needed (Generate arrays while training, increase training time - Need to compute arrays at every iteration) 
    """

    def __init__(self, joints_name=None, img_dir=None, train_data_file=None, remove_joints=None, in_size=368,
                 out_size=None):
       pass

    # --------------------Generator Initialization Methods ---------------------

    def _reduce_joints(self, joints):
        """ Select Joints of interest from self.weightJ
        """
        j = []
        for i in range(len(self.weightJ)):
            if self.weightJ[i] == 1:
                j.append(joints[2 * i])
                j.append(joints[2 * i + 1])
        return j

    def _create_train_table(self):
        raise NotImplementedError

    def _randomize(self):
        raise NotImplementedError

    def _give_batch_name(self, batch_size=16, set='train'):
        """ Returns a List of Samples
        Args:
            batch_size	: Number of sample wanted
            set				: Set to use (valid/train)
        """
        list_file = []
        for i in range(batch_size):
            if set == 'train':
                list_file.append(random.choice(self.train_set))
            elif set == 'val':
                list_file.append(random.choice(self.valid_set))
            else:
                print('Set must be : train/val')
                break
        return list_file

    def _create_sets(self, validation_rate=0.1):
       raise NotImplementedError

    def generateSet(self, rand=False):
        """ Generate the training and validation set
        Args:
            rand : (bool) True to shuffle the set
        """
        self._create_train_table()
        if rand:
            self._randomize()
        self._create_sets()

    def cwh2tlbr(self, bbox, tolist=True):
        """ Cx, Cy, W, H to TopLeft BottomRight
        """
        cx, cy, w, h = np.split(bbox, 4, axis=-1)
        x1 = cx - w/2.0
        y1 = cy - h/2.0
        x2 = cx + w/2.0
        y2 = cy + h/2.0
        box = np.concatenate([x1, y1, x2, y2], axis=-1)
        if tolist:
            return box.tolist()
        else:
            return box

    def cwh2tlbr_rev(self, bbox, tolist=True):
        """ Cx, Cy, W, H to Reversed TopLeft BottomRight
        """
        cx, cy, w, h = np.split(bbox, 4, axis=-1)
        x1 = cx - w/2.0
        y1 = cy - h/2.0
        x2 = cx + w/2.0
        y2 = cy + h/2.0
        box = np.concatenate([y1, x1, y2, x2], axis=-1)
        if tolist:
            return box.tolist()
        else:
            return box

    def tlbr2cwh(self, bbox, tolist=True):
        """ Cx, Cy, W, H to TopLeft BottomRight
        """
        x1, y1, x2, y2 = np.split(bbox, 4, axis=-1)
        w = np.abs(x2 - x1)
        h = np.abs(y2 - y1)
        cx = x1 + w/2.0
        cy = y1 + h/2.0
        box = np.concatenate([cx, cy, w, h], axis=-1)
        if tolist:
            return box.tolist()
        else:
            return box

    # ---------------------------- Generating Methods --------------------------	

    def _makeGaussian(self, height, width, sigma=3, center=None):
        """ Make a square gaussian kernel.
        size is the length of a side of the square
        sigma is full-width-half-maximum, which
        can be thought of as an effective radius.
        """
        x = np.arange(0, width, 1, float)
        y = np.arange(0, height, 1, float)[:, np.newaxis]
        if center is None:
            x0 = width // 2
            y0 = height // 2
        else:
            x0 = center[0]
            y0 = center[1]
        return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)

    def _generate_hm(self, height, width, joints, maxlength, weight, sigma_multi=1.0):
        """ Generate a full Heap Map for every joints in an array
        Args:
            height			: Wanted Height for the Heat Map
            width			: Wanted Width for the Heat Map
            joints			: Array of Joints
            maxlenght		: Lenght of the Bounding Box
        """
        num_joints = joints.shape[0]
        hm = np.zeros((height, width, num_joints + 1), dtype=np.float32)
        for i in range(num_joints):
            if not (np.array_equal(joints[i], [-1, -1])) and weight[i] == 1:
                s = int(np.sqrt(maxlength) * maxlength * 10 / 4096) + 2
                hm[:, :, i] = self._makeGaussian(height, width, sigma=s * sigma_multi, center=(joints[i, 0], joints[i, 1]))
        #   Generate background heatmap
        all_jmap = np.amax(hm[:, :, :-1], axis=-1)
        hm[:, :, -1] = 1 - all_jmap
        return hm

    def _crop_data(self, height, width, box, joints, boxp=0.05):
        """ Automatically returns a padding vector and a bounding box given
        the size of the image and a list of joints.
        Args:
            height		: Original Height
            width		: Original Width
            box			: Bounding Box
            joints		: Array o__build_phf joints
            boxp		: Box percentage (Use 20% to get a good bounding box)
        """
        padding = [[0, 0], [0, 0], [0, 0]]
        j = np.copy(joints)
        if box[0:2] == [-1, -1]:
            j[joints == -1] = 1e5
            box[0], box[1] = min(j[:, 0]), min(j[:, 1])
        crop_box = [box[0] - int(boxp * (box[2] - box[0])), box[1] - int(boxp * (box[3] - box[1])),
                    box[2] + int(boxp * (box[2] - box[0])), box[3] + int(boxp * (box[3] - box[1]))]
        if crop_box[0] < 0: crop_box[0] = 0
        if crop_box[1] < 0: crop_box[1] = 0
        if crop_box[2] > width - 1: crop_box[2] = width - 1
        if crop_box[3] > height - 1: crop_box[3] = height - 1
        new_h = int(crop_box[3] - crop_box[1])
        new_w = int(crop_box[2] - crop_box[0])
        crop_box = [crop_box[0] + new_w // 2, crop_box[1] + new_h // 2, new_w, new_h]
        if new_h > new_w:
            bounds = (crop_box[0] - new_h // 2, crop_box[0] + new_h // 2)
            if bounds[0] < 0:
                padding[1][0] = int(abs(bounds[0]))
            if bounds[1] > width - 1:
                padding[1][1] = int(abs(width - bounds[1]))
        elif new_h < new_w:
            bounds = (crop_box[1] - new_w // 2, crop_box[1] + new_w // 2)
            if bounds[0] < 0:
                padding[0][0] = int(abs(bounds[0]))
            if bounds[1] > height - 1:
                padding[0][1] = int(abs(height - bounds[1]))
        crop_box[0] += padding[1][0]
        crop_box[1] += padding[0][0]
        return padding, crop_box

    def _crop_img(self, img, padding, crop_box):
        """ Given a bounding box and padding values return cropped image
        Args:
            img			: Source Image
            padding	: Padding
            crop_box	: Bounding Box
        """
        img = np.pad(img, padding, mode='constant')
        max_lenght = max(crop_box[2], crop_box[3])
        img = img[int(crop_box[1] - max_lenght//2):int(crop_box[1] + max_lenght//2),
              int(crop_box[0]-max_lenght//2):int(crop_box[0]+max_lenght//2)]
        return img

    def _crop(self, img, hm, padding, crop_box):
        """ Given a bounding box and padding values return cropped image and heatmap
        Args:
            img			: Source Image
            hm			: Source Heat Map
            padding	: Padding
            crop_box	: Bounding Box
        """
        img = np.pad(img, padding, mode='constant')
        hm = np.pad(hm, padding, mode='constant')
        max_lenght = max(crop_box[2], crop_box[3])
        img = img[crop_box[1] - max_lenght // 2:crop_box[1] + max_lenght // 2,
              crop_box[0] - max_lenght // 2:crop_box[0] + max_lenght // 2]
        hm = hm[crop_box[1] - max_lenght // 2:crop_box[1] + max_lenght // 2,
             crop_box[0] - max_lenght // 2:crop_box[0] + max_lenght // 2]
        return img, hm

    def _relative_joints(self, box, padding, joints, to_size=64):
        """ Convert Absolute joint coordinates to crop box relative joint coordinates
        (Used to compute Heat Maps)
        Args:
            box			: Bounding Box 
            padding	: Padding Added to the original Image
            to_size	: Heat Map wanted Size
        """
        new_j = np.copy(joints)
        max_l = max(box[2], box[3])
        new_j = new_j + [padding[1][0], padding[0][0]]
        new_j = new_j - [box[0] - max_l // 2, box[1] - max_l // 2]
        new_j = new_j * (to_size / float(max_l))
        return new_j.astype(np.int32)

    def _rotate_img(self, img, r_angle):
        """ rotate augmentation
        """
        img = transform.rotate(img, r_angle, preserve_range=True)
        return img

    def _rotate_hm(self, _map, r_angle):
        """ rotate augmentation
        """
        _map[:, :, :-1] = transform.rotate(_map[:, :, :-1], r_angle)
        _map[:, :, -1] = transform.rotate(_map[:, :, -1], r_angle, cval=1)

        return _map
    
    def _rotate_mask(self, _map, r_angle):
        """ rotate augmentation
        """
        _map = transform.rotate(_map, r_angle)
        return _map
    
    def _rotate_bboxes(self, bbox_list, r_angle, hm):
        """ rotate augmentation
        """ 
        #   angle 2 radian
        radian = r_angle * math.pi / 180
        #   rotation matrix
        rotation_mat = np.array([[np.cos(radian), np.sin(radian)],
                                    [-np.sin(radian), np.cos(radian)]])
        #   rotate the bounding box
        del_ind = []
        for ind in range(bbox_list.shape[0]):
            rel_pos = np.dot(
                rotation_mat, bbox_list[ind, :2]-np.round(np.array(hm.shape[:2])/2.0))
            bbox_list[ind, :2] = rel_pos + np.round(np.array(hm.shape[:2])/2.0)
            bbox_list[ind, 2:4] = np.dot(
                np.abs(rotation_mat), bbox_list[ind, 2:4])
            if (bbox_list[ind] < 0).any() or (bbox_list[ind, :2] > hm.shape[:2]).any():
                del_ind.append(ind)
        bbox_list = np.delete(bbox_list, del_ind, 0)
        return bbox_list

    def _augment(self, img, hm, max_rotation=30):
        raise NotImplementedError
    
    def _generate_bbox(self, bbox, cbox, padd, vtype='TLBR', to_size=64, margin=0.05):
        """ convert bbox from (xmin, ymin, xmax, ymax)
            to (cx, cy, w, h)
            And transform to cbox relative position
        """
        r_bbox = -1 * np.ones((4), np.float)
        max_l = max(cbox[2], cbox[3])

        if vtype == 'TLBR':
            r_bbox = self.tlbr2cwh(np.array(bbox), tolist=False)
        elif vtype == 'CWH':
            r_bbox = bbox
        else:
            raise ValueError('Vertex point format need to be set!')
        
        r_bbox[2:4] = r_bbox[2:4] * (1+margin)

        #   Now the r_bbox is in cx, cy, w, h format
        #   transform according cbox
        #       padding offset
        r_bbox[0] = r_bbox[0] + padd[1][0]
        r_bbox[1] = r_bbox[1] + padd[0][0]
        #       crop box offset
        r_bbox[0] = r_bbox[0] - cbox[0] + max_l / 2.0
        r_bbox[1] = r_bbox[1] - cbox[1] + max_l / 2.0
        #       scaling
        r_bbox = r_bbox * (to_size / float(max_l))

        #   if bbox center is out of the image
        if r_bbox[0] < 0 or r_bbox[0] > to_size:
            return None
        if r_bbox[1] < 0 or r_bbox[1] > to_size:
            return None
        #   if the box is too large for output
        if r_bbox[2] > 2*to_size or r_bbox[3] > 2*to_size:
            return None
        return r_bbox

    def bound_judge(box_A, box_B):
        xA = max(box_A[0], box_B[0])
        yA = max(box_A[1], box_B[1])
        xB = min(box_A[2], box_B[2])
        yB = min(box_A[3], box_B[3])
        if xB-xA>0 and yB-yA>0:
            return 0
        elif box_B[0]-box_A[0]>=0 and box_B[1]-box_A[1]>=0 and box_B[2]-box_A[2]<=0 and box_B[3]-box_A[3]<=0:
            if max((box_B[2]-box_B[0]),(box_B[3]-box_B[1]))>=0:
                return 1
        elif box_B[0]-box_A[0]<=0 and box_B[1]-box_A[1]<=0 and box_B[2]-box_A[2]>=0 and box_B[3]-box_A[3]>=0:
            if max((box_B[2]-box_B[0]),(box_B[3]-box_B[1]))>=0:
                return 2
        else:
            return -1
    
    def bb_intersection_over_union(self, bboxes1, bboxes2):
        """ Numpy ndarray iou implementation
        """
        bboxes1 = np.array(bboxes1)
        bboxes2 = np.array(bboxes2)
        x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
        x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)

        # determine the (x, y)-coordinates of the intersection rectangle
        xA = np.maximum(x11, np.transpose(x21))
        yA = np.maximum(y11, np.transpose(y21))
        xB = np.minimum(x12, np.transpose(x22))
        yB = np.minimum(y12, np.transpose(y22))

        # compute the area of intersection rectangle
        interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)

        # compute the area of both the prediction and ground-truth rectangles
        boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
        boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
        iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)

        return iou
    
    def _generate_iou_map(self, width, height, bbox_list, anchors, upscale=2):
        """ generate anchor map for given anchors
        """
        #   upscale the size
        upwidth = upscale * width
        upheight = upscale * height

        iou_map = np.zeros((upheight, upwidth, len(anchors)), np.float)
        offset_map = np.zeros((upheight, upwidth, len(anchors), 4), np.float)

        bbox_list = np.array(bbox_list) * upscale
        #   convert bounding box
        bbox_list = self.cwh2tlbr(bbox_list, tolist=False)
        #   upsacle the anchor
        anchors = np.array(anchors) * upscale
        #   size * size * (x,y)
        #   construct numpy grid array
        pos = np.transpose(np.array(np.meshgrid(np.arange(upheight), np.arange(upwidth))), axes=[1,2,0])
        for bidx in range(bbox_list.shape[0]):
            for aidx in range(anchors.shape[0]):
                #   construct anchor bboxes
                anchor_bboxes = np.concatenate([pos, np.full((upheight, upwidth, 1),anchors[aidx,0]), np.full((upheight, upwidth, 1),anchors[aidx,1])], axis=-1)
                #   convert to top-left bottom-right format
                anchor_bboxes = self.cwh2tlbr(anchor_bboxes, tolist=False)
                #   reshape to dim-2 array
                anchor_bboxes = np.reshape(anchor_bboxes, (upheight * upwidth, 4))
                #   calculate boundingbox jaccard distance (IOU)
                iou = self.bb_intersection_over_union(np.expand_dims(bbox_list[bidx],axis=0), anchor_bboxes)
                #   reshape back and fill the channel
                iou_map[:, :, aidx] = np.maximum(np.reshape(iou, (upheight, upwidth)), iou_map[:, :, aidx])
                pass

        if upscale != 1:
            iou_map = msr.block_reduce(iou_map, (upscale,upscale,1), func=np.max)
            offset_map = msr.block_reduce(offset_map, (upscale,upscale,1,1), func=np.min)
        return iou_map, np.reshape(offset_map, (height, width, len(anchors)*4))

    # ----------------------- Batch Generator ----------------------------------

    def _aux_generator(self, batch_size=16, stacks=4, normalize=True, sample_set='train'):
        raise NotImplementedError

    def generator(self, *args, **kwargs):
        """ Create a Sample Generator
        Args:
            batchSize 	: Number of image per batch 
            stacks 	 	: Stacks in HG model
            norm 	 	 	: (bool) True to normalize the batch
            sample 	 	: 'train'/'val' Default: 'train'
        """
        return self._aux_generator(*args, **kwargs)

    # ---------------------------- Image Reader --------------------------------				
    def open_img(self, name, color='RGB'):
        """ Open an image 
        Args:
            name	: Name of the sample
            color	: Color Mode (RGB/BGR/GRAY)
        """
        if name[-1] in self.letter:
            name = name[:-1]
        img = cv2.imread(os.path.join(self.img_dir, name))
        if color == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        elif color == 'BGR':
            return img
        elif color == 'GRAY':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            print('Color mode supported: RGB/BGR. If you need another mode do it yourself :p')

    def plot_img(self, name, plot='cv2'):
        """ Plot an image
        Args:
            name	: Name of the Sample
            plot	: Library to use (cv2: OpenCV, plt: matplotlib)
        """
        if plot == 'cv2':
            img = self.open_img(name, color='BGR')
            cv2.imshow('Image', img)
        elif plot == 'plt':
            img = self.open_img(name, color='RGB')
            plt.imshow(img)
            plt.show()

    def test(self, toWait=0.2):
        """ TESTING METHOD
        You can run it to see if the preprocessing is well done.
        Wait few seconds for loading, then diaporama appears with image and highlighted joints
        /!\ Use Esc to quit
        Args:
            toWait : In sec, time between pictures
        """
        self._create_train_table()
        self._create_sets()
        for i in range(len(self.train_set)):
            img = self.open_img(self.train_set[i])
            w = self.data_dict[self.train_set[i]]['weights']
            padd, box = self._crop_data(img.shape[0], img.shape[1], self.data_dict[self.train_set[i]]['box'],
                                        self.data_dict[self.train_set[i]]['joints'], boxp=0.0)
            new_j = self._relative_joints(box, padd, self.data_dict[self.train_set[i]]['joints'], to_size=self.in_size)
            rhm = self._generate_hm(self.in_size, self.in_size, new_j, self.in_size, w)
            rimg = self._crop_img(img, padd, box)
            # See Error in self._generator
            # rimg = cv2.resize(rimg, (self.in_size,self.in_size))
            rimg = scm.imresize(rimg, (self.in_size, self.in_size))
            # rhm = np.zeros((self.in_size,self.in_size,16))
            # for i in range(16):
            #	rhm[:,:,i] = cv2.resize(rHM[:,:,i], (self.in_size,self.in_size))
            grimg = cv2.cvtColor(rimg, cv2.COLOR_RGB2GRAY)
            cv2.imshow('image', grimg / 255 + np.sum(rhm, axis=2))
            # Wait
            time.sleep(toWait)
            if cv2.waitKey(1) == 27:
                print('Ended')
                cv2.destroyAllWindows()
                break

    # ------------------------------- PCK METHODS-------------------------------
    def pck_ready(self, idlh=3, idrs=12, testSet=None):
        """ Creates a list with all PCK ready samples
        (PCK: Percentage of Correct Keypoints)
        """
        id_lhip = idlh
        id_rsho = idrs
        self.total_joints = 0
        self.pck_samples = []
        for s in self.data_dict.keys():
            if testSet == None:
                if self.data_dict[s]['weights'][id_lhip] == 1 and self.data_dict[s]['weights'][id_rsho] == 1:
                    self.pck_samples.append(s)
                    wIntel = np.unique(self.data_dict[s]['weights'], return_counts=True)
                    self.total_joints += dict(zip(wIntel[0], wIntel[1]))[1]
            else:
                if self.data_dict[s]['weights'][id_lhip] == 1 and self.data_dict[s]['weights'][
                    id_rsho] == 1 and s in testSet:
                    self.pck_samples.append(s)
                    wIntel = np.unique(self.data_dict[s]['weights'], return_counts=True)
                    self.total_joints += dict(zip(wIntel[0], wIntel[1]))[1]
        print('PCK PREPROCESS DONE: \n --Samples:', len(self.pck_samples), '\n --Num.Joints', self.total_joints)

    def getSample(self, sample=None):
        """ Returns information of a sample
        Args:
            sample : (str) Name of the sample
        Returns:
            img: RGB Image
            new_j: Resized Joints 
            w: Weights of Joints
            joint_full: Raw Joints
            max_l: Maximum Size of Input Image
        """
        if sample != None:
            joints = self.data_dict[sample]['joints']
            box = self.data_dict[sample]['box']
            w = self.data_dict[sample]['weights']
            img = self.open_img(sample)
            padd, cbox = self._crop_data(img.shape[0], img.shape[1], box, joints, boxp=0.2)
            new_j = self._relative_joints(cbox, padd, joints, to_size=self.in_size)
            joint_full = np.copy(joints)
            max_l = max(cbox[2], cbox[3])
            joint_full = joint_full + [padd[1][0], padd[0][0]]
            joint_full = joint_full - [cbox[0] - max_l // 2, cbox[1] - max_l // 2]
            img = self._crop_img(img, padd, cbox)
            img = img.astype(np.uint8)
            img = scm.imresize(img, (self.in_size, self.in_size))
            return img, new_j, w, joint_full, max_l
        else:
            print('Specify a sample name')

if __name__ == '__main__':
    gen = DataGenerator()
    stride = 2
    padding = 0
    shape = 368
    scale = 8
    size = scale * scale
    X = cv2.imread("../test.1.png")
    X = cv2.resize(X, (shape,shape))
    cv2.imwrite('shape.jpg', X)
    out = msr.block_reduce(X, (scale,scale,1), func=np.max)
    cv2.imwrite("out.jpg", out)