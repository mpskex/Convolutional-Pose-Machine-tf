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
"""
import os
import time
import numpy as np
import tensorflow as tf
import PoseNet

class CPM(PoseNet.PoseNet):
    """
    CPM net
    """
    def __init__(self, base_lr=0.0005, in_size=368, out_size=None, batch_size=16, epoch=20, dataset = None, log_dir=None, stage=6,
                 epoch_size=1000, w_summary=True, training=True, joints=None, cpu_only=False, pretrained_model='vgg19.npy',
                 load_pretrained=False, predict=False):
        """ CPM Net implemented with Tensorflow

        :param base_lr:             starter learning rate
        :param in_size:             input image size
        :param batch_size:          size of each batch
        :param epoch:               num of epoch to train
        :param dataset:             *datagen* class to gen & feed data
        :param log_dir:             log directory
        :param stage:               num of stage in cpm model
        :param epoch_size:          size of each epoch
        :param w_summary:           bool to determine if do weight summary
        :param training:            bool to determine if the model trains
        :param joints:              list to define names of joints
        :param cpu_only:            CPU mode or GPU mode
        :param pretrained_model:    Path to pre-trained model
        :param load_pretrained:     bool to determine if the net loads all arg

        ATTENTION HERE:
        *   if load_pretrained is False
            then the model only loads VGG part of arguments
            if true, then it loads all weights & bias

        *   if log_dir is None, then the model won't output any save files
            but PLEASE DONT WORRY, we defines a default log ditectory

        TODO:
            *   Save model as numpy
            *   Predicting codes
            *   PCKh & mAP Test code
        """
        tf.reset_default_graph()
        self.sess = tf.Session()

        #   model log dir control
        if log_dir is not None:
            self.writer = tf.summary.FileWriter(log_dir)
            self.log_dir = log_dir
        else:
            self.log_dir = 'log/'

        #   model device control
        self.cpu = '/cpu:0'
        if cpu_only:
            self.gpu = self.cpu
        else:
            self.gpu = '/gpu:0'

        self.dataset = dataset

        #   Annotations Associated
        if joints is not None:
            self.joints = joints
        else:
            self.joints = ['r_anckle', 'r_knee', 'r_hip', 'l_hip', 'l_knee', 'l_anckle', 'pelvis', 'thorax', 'neck', 'head', 'r_wrist', 'r_elbow', 'r_shoulder', 'l_sho    ulder', 'l_elbow', 'l_wrist']
        self.joint_num = len(self.joints)

        #   Net Args
        self.stage = stage
        self.training = training
        self.base_lr = base_lr
        self.in_size = in_size
        if out_size is None:
            self.out_size = self.in_size/8
        else:
            self.out_size = out_size
        self.batch_size = batch_size
        self.epoch = epoch
        self.epoch_size = epoch_size
        self.dataset = dataset

        #   step learning rate policy
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(base_lr,
            self.global_step, 10*self.epoch*self.epoch_size, 0.333,
            staircase=True)

        #   Inside Variable
        self.train_step = []
        self.losses = []
        self.w_summary = w_summary
        self.net_debug = False

        self.img = None
        self.gtmap = None

        self.summ_scalar_list = []
        self.summ_accuracy_list = []
        self.summ_image_list = []
        self.summ_histogram_list = []

        #   load model
        self.load_pretrained = load_pretrained
        if pretrained_model is not None:
            self.pretrained_model = np.load(pretrained_model, encoding='latin1').item()
            print("[*]\tnumpy file loaded!")
        else:
            self.pretrained_model = None

        #   dictionary of network parameters
        self.var_dict = {}



    def net(self, image, name='CPM'):
        """ CPM Net Structure
        Args:
            image           : Input image with n times of 8
                                size:   batch_size * in_size * in_size * sizeof(RGB)
        Return:
            stacked heatmap : Heatmap NSHWC format
                                size:   batch_size * stage_num * in_size/8 * in_size/8 * joint_num
        """
        with tf.variable_scope(name):
            fmap = self._feature_extractor(image, 'VGG', 'Feature_Extractor')
            stage = [None] * self.stage
            stage[0] = self._cpm_stage(fmap, 1, None)
            for t in range(2,self.stage+1):
                stage[t-1] = self._cpm_stage(fmap, t, stage[t-2])
            #   RETURN SIZE:
            #       batch_size * stage_num * in_size/8 * in_size/8 * joint_num
            return tf.nn.sigmoid(tf.stack(stage, axis= 1 , name= 'stack_output'),name = 'final_output')

    def _feature_extractor(self, inputs, net_type='VGG', name='Feature_Extractor'):
        """ Feature Extractor
        For VGG Feature Extractor down-scale by x8
        For ResNet Feature Extractor downscale by x8 (Current Setup)

        Net use VGG as default setup
        Args:
            inputs      : Input Tensor (Data Format: NHWC)
            name        : Name of the Extractor
        Returns:
            net         : Output Tensor            
        """
        with tf.variable_scope(name):
            if net_type == 'ResNet':
                net = self._conv_bn_relu(inputs, 64, 7, 2, 'SAME')
                #   down scale by 2
                net = self._residual(net, 128, 'r1')
                net = tf.contrib.layers.max_pool2d(net, [2,2], [2,2], padding='SAME', name='pool1')
                #   down scale by 2
                net = self._residual(net, 128, 'r2')
                net = self._residual(net, 256, 'r3')
                net = tf.contrib.layers.max_pool2d(net, [2,2], [2,2], padding='SAME', name='pool2')
                #   down scale by 2
                net = tf.contrib.layers.max_pool2d(net, [2,2], [2,2], padding='SAME', name='pool3')
                net = self._residual(net, 512, 'r4')
                net = tf.contrib.layers.max_pool2d(net, [2,2], [2,2], padding='SAME', name='pool4')
                #   optional 
                #net = self._residual(net, 512, 'r5')
                return net
            else:
                #   VGG based
                net = self._conv_bn_relu(inputs, 64, 3, 1, 'SAME', 'conv1_1', use_loaded=True, lock=True)
                net = self._conv_bn_relu(net, 64, 3, 1, 'SAME', 'conv1_2', use_loaded=True, lock=True)
                net = tf.contrib.layers.max_pool2d(net, [2,2], [2,2], padding='SAME', scope='pool1')
                #   down scale by 2
                net = self._conv_bn_relu(net, 128, 3, 1, 'SAME', 'conv2_1', use_loaded=True, lock=True)
                net = self._conv_bn_relu(net, 128, 3, 1, 'SAME', 'conv2_2', use_loaded=True, lock=True)
                net = tf.contrib.layers.max_pool2d(net, [2,2], [2,2], padding='SAME', scope='pool2')
                #   down scale by 2
                net = self._conv_bn_relu(net, 256, 3, 1, 'SAME', 'conv3_1', use_loaded=True, lock=True)
                net = self._conv_bn_relu(net, 256, 3, 1, 'SAME', 'conv3_2', use_loaded=True, lock=True)
                net = self._conv_bn_relu(net, 256, 3, 1, 'SAME', 'conv3_3', use_loaded=True, lock=True)
                net = self._conv_bn_relu(net, 256, 3, 1, 'SAME', 'conv3_4', use_loaded=True, lock=True)
                net = tf.contrib.layers.max_pool2d(net, [2,2], [2,2], padding='SAME', scope='pool3')
                #   down scale by 2
                net = self._conv_bn_relu(net, 512, 3, 1, 'SAME', 'conv4_1', use_loaded=True, lock=True)
                net = self._conv_bn_relu(net, 512, 3, 1, 'SAME', 'conv4_2', use_loaded=True, lock=True)
                return net

    def _cpm_stage(self, feat_map, stage_num, last_stage = None):
        """ CPM stage Sturcture
        Args:
            feat_map    : Input Tensor from feature extractor
            last_stage  : Input Tensor from below
            stage_num   : stage number
            name        : name of the stage
        """
        with tf.variable_scope('CPM_stage'+str(stage_num)):
            if stage_num == 1:
                net = self._conv_bn_relu(feat_map, 256, 3, 1, 'SAME', 'conv4_3_CPM', use_loaded=self.load_pretrained, lock=not self.training)
                net = self._conv_bn_relu(net, 256, 3, 1, 'SAME', 'conv4_4_CPM', use_loaded=self.load_pretrained, lock=not self.training)
                net = self._conv_bn_relu(net, 256, 3, 1, 'SAME', 'conv4_5_CPM', use_loaded=self.load_pretrained, lock=not self.training)
                net = self._conv_bn_relu(net, 256, 3, 1, 'SAME', 'conv4_6_CPM', use_loaded=self.load_pretrained, lock=not self.training)
                net = self._conv_bn_relu(net, 128, 3, 1, 'SAME', 'conv4_7_CPM', use_loaded=self.load_pretrained, lock=not self.training)
                net = self._conv_bn_relu(net, 512, 1, 1, 'SAME', 'conv5_1_CPM', use_loaded=self.load_pretrained, lock=not self.training)
                net = self._conv(net, self.joint_num+1, 1, 1, 'SAME', 'conv5_2_CPM', use_loaded=self.load_pretrained, lock=not self.training)
                return net
            elif stage_num > 1:
                net = tf.concat([feat_map, last_stage], 3)
                net = self._conv_bn_relu(net, 128, 7, 1, 'SAME', 'Mconv1_stage'+str(stage_num), use_loaded=self.load_pretrained, lock=not self.training)
                net = self._conv_bn_relu(net, 128, 7, 1, 'SAME', 'Mconv2_stage'+str(stage_num), use_loaded=self.load_pretrained, lock=not self.training)
                net = self._conv_bn_relu(net, 128, 7, 1, 'SAME', 'Mconv3_stage'+str(stage_num), use_loaded=self.load_pretrained, lock=not self.training)
                net = self._conv_bn_relu(net, 128, 7, 1, 'SAME', 'Mconv4_stage'+str(stage_num), use_loaded=self.load_pretrained, lock=not self.training)
                net = self._conv_bn_relu(net, 128, 7, 1, 'SAME', 'Mconv5_stage'+str(stage_num), use_loaded=self.load_pretrained, lock=not self.training)
                net = self._conv_bn_relu(net, 128, 1, 1, 'SAME', 'Mconv6_stage'+str(stage_num), use_loaded=self.load_pretrained, lock=not self.training)
                net = self._conv(net, self.joint_num+1, 1, 1, 'SAME', 'Mconv7_stage'+str(stage_num), use_loaded=self.load_pretrained, lock=not self.training)
                return net

