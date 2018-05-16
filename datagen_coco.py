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


class DataGenerator(object):
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

        if COCO_anno_file_val is not None:
            self.coco_anno_val = COCO(COCO_anno_file_val)
            catIds = self.coco_anno_val.getCatIds(catNms=['person'])
            imgIds = self.coco_anno_val.getImgIds(catIds=catIds)
            annIds = self.coco_anno_val.getAnnIds(catIds=catIds)
            self.anno_list_val = self.coco_anno_val.loadAnns(annIds)
        else:
            self.coco_anno_val = None
            self.anno_list_val = None
            print '[!]\tNo Validation Set is Found! The monitoring data might be accurate!'
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

    def _complete_sample(self, name):
        """ Check if a sample has no missing value
        Args:
            name 	: Name of the sample
        """
        raise NotImplementedError

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
            if (joints==-1).all():
                #print "[*]\tWarning:\tNull data in COCO!"
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
                if (joints==-1).all():
                    #print "[*]\tWarning:\tNull data in COCO!"
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

    def _generate_hm(self, height, width, joints, maxlenght, weight):
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
                s = int(np.sqrt(maxlenght) * maxlenght * 10 / 4096) + 2
                hm[:, :, i] = self._makeGaussian(height, width, sigma=s, center=(joints[i, 0], joints[i, 1]))
            else:
                hm[:, :, i] = np.zeros((height, width))
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
            if bounds[1] > width - 1:
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

    def _crop(self, img, hm, padding, crop_box):
        """ Given a bounding box and padding values return cropped image and heatmap
        Args:
            img			: Source Imageann['image_id']]
        h, w = t['height'], t['width'
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
        new_j = new_j * to_size / (max_l + 0.0000001)
        return new_j.astype(np.int32)

    def _augment(self, img, hm, mask, max_rotation=30, angle=None):
        """ # TODO : IMPLEMENT DATA AUGMENTATION 
        """
        if random.choice([0, 1]):
            if angle==None:
                r_angle = np.random.randint(-1 * max_rotation, max_rotation)
            else:
                r_angle = angle
            img = transform.rotate(img, r_angle, preserve_range=True)
            mask = transform.rotate(mask, r_angle, preserve_range=True)
            hm = transform.rotate(hm, r_angle)
        return img, hm, mask

    # ----------------------- Batch Generator ----------------------------------

    def _aux_generator(self, batch_size=16, stacks=4, normalize=True, sample_set='train', debug=False):
        """ Auxiliary Generator
        Args:
            See Args section in self._generator
        """
        while True:
            train_img = np.zeros((batch_size, self.in_size, self.in_size, 3), dtype=np.float32)
            train_gtmap = np.zeros((batch_size, stacks, self.out_size, self.out_size, self.joints_num+1), np.float32)
            train_weights = np.zeros((batch_size, self.joints_num + 1), np.float32)
            train_mask = np.ones((batch_size, stacks, self.out_size, self.out_size, self.joints_num+1), np.float32)
            i = 0
            while i < batch_size:
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
                if (joints==-1).all():
                    print "[!]\tError:\tDetecting null points!"
                    continue
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
                mask = np.expand_dims(mask, axis=0)
                mask = np.repeat(mask, self.joints_num+1, axis=0)
                mask = np.expand_dims(mask, axis=0)
                mask = np.repeat(mask, stacks, axis=0)
                mask = np.transpose(mask, axes=[0,2,3,1])
                hm = np.expand_dims(hm, axis=0)
                hm = np.repeat(hm, stacks, axis=0)
                train_weights[i][:self.joints_num] = weight
                if normalize:
                    train_img[i] = img.astype(np.float32) / 255
                else:
                    train_img[i] = img.astype(np.float32)
                train_gtmap[i] = hm
                train_mask[i] = mask
                i = i + 1
                '''
                except:
                    print "[!]\tError while generating the batch!"
                '''
            #print train_weights
            yield train_img, train_gtmap, train_weights, train_mask

    def generator(self, batchSize=16, stacks=4, norm=True, sample_set='train', debug=False):
        """ Create a Sample Generator
        Args:
            batchSize 	: Number of image per batch 
            stacks 	 	: Stacks in HG model
            norm 	 	 	: (bool) True to normalize the batch
            sample 	 	: 'train'/'valid' Default: 'train'
        """
        return self._aux_generator(batch_size=batchSize, stacks=stacks, normalize=norm, sample_set=sample_set, debug=debug)

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
        self.generator

    # ------------------------------- PCK METHODS-------------------------------
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
            try:
                joints = self.data_dict[sample]['keypoints']
                box = self.data_dict[sample]['bbox']
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
            except:
                print('\t\tError!! in getSample()')
                raise RuntimeError
        else:
            print('Specify a sample name')

if __name__ == '__main__':
    #   module testing code
    INPUT_SIZE = 368
    IMG_ROOT = "/home/mpsk/data/COCO2017/val2017"
    COCO_anno_file = "/home/mpsk/data/COCO2017/annotations/person_keypoints_val2017.json"
    gen = DataGenerator(img_dir=IMG_ROOT, COCO_anno_file=COCO_anno_file, in_size=INPUT_SIZE)
    gen.generateSet(rand=True)
    for n in range(1000):
        img, gtmap, w, mask = next(gen.generator(debug=True))
        for i in (img, gtmap, w, mask):
            print i.shape
        io.imsave('test.img.jpg', img[0])
        io.imsave("test.mask.jpg", mask[0])
        print np.average(mask)
        break
