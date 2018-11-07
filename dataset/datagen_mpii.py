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
import matplotlib.pyplot as plt
import random
import time
from skimage import transform
import scipy.misc as scm
import skimage.io as io
import datagen

class DataGenerator(datagen.DataGenerator):
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
        """ Initializer
        Args:
            joints_name			: List of joints condsidered
            img_dir				: Directory containing every images
            train_data_file		: Text file with training set data
            remove_joints		: Joints List to keep (See documentation)
        """
        if joints_name == None:
            self.joints_list = ['r_anckle', 'r_knee', 'r_hip', 'l_hip', 'l_knee', 'l_anckle', 'pelvis', 'thorax',
                                'neck', 'head', 'r_wrist', 'r_elbow', 'r_shoulder', 'l_shoulder', 'l_elbow', 'l_wrist']
        else:
            self.joints_list = joints_name
        self.toReduce = False
        if remove_joints is not None:
            self.toReduce = True
            self.weightJ = remove_joints

        self.in_size = in_size
        if out_size is None:
            self.out_size = self.in_size / 8
        else:
            self.out_size = out_size
        self.joints_num = len(self.joints_list)

        self.letter = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
        self.img_dir = img_dir
        self.train_data_file = train_data_file
        self.images = os.listdir(img_dir)
        self.name = 'mpii'

    # --------------------Generator Initialization Methods ---------------------

    def _create_train_table(self):
        """ Create Table of samples from TEXT file
        """
        self.train_table = []
        self.no_intel = []
        self.data_dict = {}
        input_file = open(self.train_data_file, 'r')
        print('READING TRAIN DATA')
        for line in input_file:
            line = line.strip()
            line = line.split(' ')
            name = line[0]
            box = list(map(int, line[1:5]))
            joints = list(map(int, line[5:]))
            if self.toReduce:
                joints = self._reduce_joints(joints)
            if joints == [-1] * len(joints):
                self.no_intel.append(name)
            else:
                joints = np.reshape(joints, (-1, 2))
                w = [1] * joints.shape[0]
                for i in range(joints.shape[0]):
                    if np.array_equal(joints[i], [-1, -1]):
                        w[i] = 0
                self.data_dict[name] = {'box': box, 'joints': joints, 'weights': w}
                self.train_table.append(name)
        input_file.close()

    def _randomize(self):
        """ Randomize the set
        """
        random.shuffle(self.train_table)

    def _complete_sample(self, name):
        """ Check if a sample has no missing value
        Args:
            name 	: Name of the sample
        """
        for i in range(self.data_dict[name]['joints'].shape[0]):
            if np.array_equal(self.data_dict[name]['joints'][i], [-1, -1]):
                return False
        return True

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
        """ Select Elements to feed training and validation set 
        Args:
            validation_rate		: Percentage of validation data (in ]0,1[, don't waste time use 0.1)
        """
        sample = len(self.train_table)
        valid_sample = int(sample * validation_rate)
        self.train_set = self.train_table[:sample - valid_sample]
        self.valid_set = []
        preset = self.train_table[sample - valid_sample:]
        print('START SET CREATION')
        for elem in preset:
            if self._complete_sample(elem):
                self.valid_set.append(elem)
            else:
                self.train_set.append(elem)
        print('SET CREATED')
        np.save('Dataset-Validation-Set', self.valid_set)
        np.save('Dataset-Training-Set', self.train_set)
        print('--Training set :', len(self.train_set), ' samples.')
        print('--Validation set :', len(self.valid_set), ' samples.')

    def generateSet(self, rand=False):
        """ Generate the training and validation set
        Args:
            rand : (bool) True to shuffle the set
        """
        self._create_train_table()
        if rand:
            self._randomize()
        self._create_sets()

    def _augment(self, img, hm, max_rotation=30):
        """ # TODO : IMPLEMENT DATA AUGMENTATION 
        """
        if random.choice([0, 1]):
            r_angle = np.random.randint(-1 * max_rotation, max_rotation)
            '''
            #   old version
            img = transform.rotate(img, r_angle, preserve_range=True)
            hm = transform.rotate(hm, r_angle)
            '''
            img = self._rotate_img(img, r_angle)
            hm = self._rotate_hm(hm, r_angle)
        return img, hm

    # ----------------------- Batch Generator ----------------------------------

    def _aux_generator(self, batch_size=16, stacks=4, normalize=True, sample_set='train'):
        """ Auxiliary Generator
        Args:
            See Args section in self.generator
        """
        while True:
            train_img = np.zeros((batch_size, self.in_size, self.in_size, 3), dtype=np.float32)
            train_gtmap = np.zeros((batch_size, stacks, self.out_size, self.out_size, len(self.joints_list) + 1),
                                   np.float32)
            train_weights = np.zeros((batch_size, len(self.joints_list) + 1), np.float32)
            train_mask = np.zeros((batch_size, self.out_size, self.out_size, len(self.joints_list) + 1), np.float32)
            i = 0
            while i < batch_size:
                try:
                    if sample_set == 'train':
                        name = random.choice(self.train_set)
                    elif sample_set == 'val':
                        name = random.choice(self.valid_set)
                    joints = self.data_dict[name]['joints']
                    box = self.data_dict[name]['box']
                    weight = np.asarray(self.data_dict[name]['weights'])
                    train_weights[i][:len(self.joints_list)] = weight
                    img = self.open_img(name)
                    padd, cbox = self._crop_data(img.shape[0], img.shape[1], box, joints, boxp=0.2)
                    new_j = self._relative_joints(cbox, padd, joints, to_size=self.out_size)
                    hm = self._generate_hm(self.out_size, self.out_size, new_j, self.out_size, weight)
                    img = self._crop_img(img, padd, cbox)
                    img = img.astype(np.uint8)
                    img = scm.imresize(img, (self.in_size, self.in_size))
                    img, hm = self._augment(img, hm)
                    hm = np.expand_dims(hm, axis=0)
                    hm = np.repeat(hm, stacks, axis=0)
                    if normalize:
                        train_img[i] = img.astype(np.float32) / 255
                    else:
                        train_img[i] = img.astype(np.float32)
                    train_gtmap[i] = hm
                    i = i + 1
                except Exception, e:
                    print 'error file: ', name, ' Info: ', str(e)
            yield train_img, train_gtmap, train_weights

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
    # module testing code
    INPUT_SIZE = 368
    # IMG_ROOT = "/Users/mpsk/Documents/COCO2017/val2017"
    # COCO_anno_file = "/Users/mpsk/Documents/COCO2017/annotations/person_keypoints_val2017.json"
    IMG_ROOT = "/var/data/Dataset/images"
    gen = DataGenerator(img_dir=IMG_ROOT, train_data_file='../dataset.txt', in_size=INPUT_SIZE)
    gen.generateSet(rand=True)
    img, gtmap, w = next(gen.generator())
    io.imsave('1.jpg', img[0])
    io.imsave('p.jpg', (gtmap[0,0,:,:,0]*255).astype(np.uint8))
    print w
