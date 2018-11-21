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
import Regularizer

class CPM(PoseNet.PoseNet):
    """
    CPM net
    """
    def __init__(self, base_lr=0.0005, in_size=368, out_size=None, batch_size=16, epoch=20, dataset = None, log_dir=None, stage=6,
                 epoch_size=1000, w_summary=True, training=True, joints=None, cpu_only=False, pretrained_model='vgg19.npy',
                 load_pretrained=False, predict=False, name='CPM'):
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

        #   model log dir control
        self.log_dir = name + '_lr' + str(base_lr) + '_insize' + str(in_size) + '/'
        self.name = name
        print "[*]\tLog dir is : ", self.log_dir


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
            self.joints = ['r_anckle', 'r_knee', 'r_hip', 'l_hip', 'l_knee', 'l_anckle', 'pelvis', 'thorax', 'neck', 'head', 'r_wrist', 'r_elbow', 'r_shoulder', 'l_shoulder', 'l_elbow', 'l_wrist']
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
        #   r_mask = mask_level + (1 - mask_level) * mask
        self.mask_level = 0.2

        #   step learning rate policy
        self.base_lr = base_lr

        #   Inside Variable
        self.train_step = []
        self.losses = []
        self.w_summary = w_summary
        self.net_debug = False

        self.img = None
        self.joint_map_gt = None

        self.summ_scalar_list = []
        self.summ_accuracy_list = []
        self.summ_image_list = []
        self.summ_histogram_list = []

        #   load model
        self.load_pretrained = load_pretrained
        if pretrained_model is not None:
            self.pretrained_model = np.load(pretrained_model, encoding='latin1').item()
            print("[*]\tPretrained File params loaded!")
        else:
            self.pretrained_model = None

        #   dictionary of network parameters
        self.var_dict = {}

    def build_ph(self):
        """ Building Placeholder in tensorflow session
        :return:
        """
        #   Valid & Train input
        #   input image : channel 3
        self.img = tf.placeholder(tf.float32, 
            shape=[None, self.in_size, self.in_size, 3], name="img_in")
        #   input center map : channel 1 (downscale by 8)
        self.joint_weight = tf.placeholder(tf.float32 ,
            shape=[None, self.joint_num+1])

        #   Train input
        #   input ground truth : channel 1 (downscale by 8)
        self.joint_map_gt = tf.placeholder(tf.float32, 
            shape=[None, self.stage, self.out_size, self.out_size, self.joint_num+1], name="joint_map_gt")

    def build_train_op(self):
        """ Building training associates: losses & loss summary
        :return:
        """
        #   Optimizer
        with tf.name_scope('loss'):
            with tf.variable_scope("JointLoss"):
                loss = tf.multiply(self.joint_weight, tf.reduce_sum(tf.nn.l2_loss(self.joint_map - self.joint_map_gt)))
            with tf.variable_scope("Regularization"):
                self.norm = []
                for regularizer in self.regularizers:
                    self.norm.append(regularizer())
            self.losses.append(loss)
            self.total_loss = tf.reduce_mean(self.losses) + tf.reduce_sum(self.norm)
            self.summ_scalar_list.append(tf.summary.scalar("total loss", self.total_loss))
            self.summ_scalar_list.append(tf.summary.scalar("lr", self.learning_rate))
            print("- LOSS & SCALAR_SUMMARY build finished!")
        with tf.name_scope('optimizer'):
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
                #self.optimizer = tf.train.AdamOptimizer(self.learning_rate, epsilon=1e-8)
                #   Global train
                self.train_step.append(self.optimizer.minimize(self.total_loss/self.batch_size,
                                                                global_step=self.global_step))
        print("- OPTIMIZER build finished!")


    def BuildModel(self, debug=False):
        """ Building model in tensorflow session

        :return:
        """
        #   input
        tf.reset_default_graph()
        self.sess = tf.Session()
        if self.training:
            self.writer = tf.summary.FileWriter(self.log_dir)
            self.regularizers = [Regularizer.L2Regularizer(beta=1e-4)]
            self.global_step = tf.Variable(0, trainable=False)
            self.learning_rate = tf.train.exponential_decay(self.base_lr,
                self.global_step, 50*self.epoch_size, 0.333,
                staircase=True)
        with tf.name_scope('input'):
            self.build_ph()
        #   assertion
        assert self.img!=None and self.joint_map_gt!=None
        self.joint_map = self.net(self.img)
        if not debug:
            #   the net
            if self.training:
                #   train op
                with tf.name_scope('train'):
                    self.build_train_op()
                with tf.name_scope('image_summary'):
                    self.build_monitor()
                with tf.name_scope('accuracy'):
                    self.build_accuracy()
            #   initialize all variables
            self.sess.run(tf.global_variables_initializer())
            if self.training:
                self.saver = tf.train.Saver()
                #   merge all summary
                self.summ_image = tf.summary.merge(self.summ_image_list)
                self.summ_scalar = tf.summary.merge(self.summ_scalar_list)
                self.summ_accuracy = tf.summary.merge(self.summ_accuracy_list)
                self.summ_histogram = tf.summary.merge(self.summ_histogram_list)
                self.writer.add_graph(self.sess.graph)
        print("[*]\tModel Built")

    def train(self):
        """ Training Progress in CPM

        :return:    Nothing to output
        """
        _epoch_count = 0
        _iter_count = 0
    
        #   datagen from Hourglass
        self.generator = self.dataset.generator(self.batch_size, stacks=self.stage, norm=True, sample_set='train')
        self.valid_gen = self.dataset.generator(self.batch_size, stacks=self.stage, norm=True, sample_set='val')

        for n in range(self.epoch):
            for m in range(self.epoch_size):
                #   datagen from hourglass
                _train_batch = next(self.generator)
                print("[*] small batch generated!")
                for step in self.train_step:
                    self.sess.run(step, feed_dict={self.img: _train_batch[0],
                                                    self.joint_map_gt:_train_batch[1],
                                                    self.joint_weight:_train_batch[2]})
                if _iter_count % 10 == 0:
                    _test_batch = next(self.valid_gen)
                    #   doing the scalar summary
                    summ_scalar_out, summ_img_out, summ_acc_out, summ_hist_out, jloss_out = self.sess.run(
                                [self.summ_scalar, self.summ_image, self.summ_accuracy, self.summ_histogram, self.total_loss],
                                                    feed_dict={self.img: _test_batch[0],
                                                                self.joint_map_gt: _test_batch[1],
                                                                self.joint_weight:_test_batch[2]})
                    for summ in [summ_scalar_out, summ_img_out, summ_acc_out, summ_hist_out]:
                        self.writer.add_summary(summ, _iter_count)
                    print("epoch ", _epoch_count, " iter ", _iter_count, [jloss_out])

                if _iter_count % 500 == 0:
                    #   generate heatmap from the network
                    maps = self.sess.run(self.joint_map,
                            feed_dict={self.img: _test_batch[0],
                                    self.joint_map_gt: _test_batch[1],
                                    self.joint_weight: _test_batch[2]})
                    if self.log_dir is not None:
                        print("[!] saved heatmap with size of ", maps.shape)
                        np.save(self.log_dir+"output.npy", maps)
                        print("[!] saved ground truth with size of ", self.joint_map_gt.shape)
                        np.save(self.log_dir+"gt.npy", _test_batch[1])
                    del maps, _test_batch
                print("iter:", _iter_count)
                _iter_count += 1
                self.writer.flush()
                del _train_batch
            #   doing save numpy params
            self.save_npy()
            _epoch_count += 1
            #   save model every epoch
            if self.log_dir is not None:
                self.saver.save(os.path.join(self.log_dir, "model.ckpt"), n)

    def net(self, image, name='CPM', load_pretrained=False, lock=False):
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
            stage[0], _ = self._cpm_stage(fmap, 1, None, load_pretrained= self.load_pretrained, lock=not self.training)
            for t in range(2,self.stage+1):
                stage[t-1], _ = self._cpm_stage(fmap, t, stage[t-2], load_pretrained=self.load_pretrained, lock= not self.training)
            #   RETURN SIZE:
            #       batch_size * stage_num * in_size/8 * in_size/8 * joint_num
            return tf.nn.sigmoid(tf.stack(stage, axis=1 , name= 'stack_output'),name = 'final_output')

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
        if net_type == 'ResNet':
            with tf.variable_scope(net_type):
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
        elif net_type == 'VGG':
            with tf.variable_scope(net_type):
                #   VGG based
                net = self._conv_bias_relu(inputs, 64, 3, 1, 'SAME', 'conv1_1', use_loaded=True, lock=True)
                net = self._conv_bias_relu(net, 64, 3, 1, 'SAME', 'conv1_2', use_loaded=True, lock=True)
                net = tf.contrib.layers.max_pool2d(net, [2,2], [2,2], padding='SAME', scope='pool1')
                #   down scale by 2
                net = self._conv_bias_relu(net, 128, 3, 1, 'SAME', 'conv2_1', use_loaded=True, lock=True)
                net = self._conv_bias_relu(net, 128, 3, 1, 'SAME', 'conv2_2', use_loaded=True, lock=True)
                net = tf.contrib.layers.max_pool2d(net, [2,2], [2,2], padding='SAME', scope='pool2')
                #   down scale by 2
                net = self._conv_bias_relu(net, 256, 3, 1, 'SAME', 'conv3_1', use_loaded=True, lock=True)
                net = self._conv_bias_relu(net, 256, 3, 1, 'SAME', 'conv3_2', use_loaded=True, lock=True)
                net = self._conv_bias_relu(net, 256, 3, 1, 'SAME', 'conv3_3', use_loaded=True, lock=True)
                net = self._conv_bias_relu(net, 256, 3, 1, 'SAME', 'conv3_4', use_loaded=True, lock=True)
                net = tf.contrib.layers.max_pool2d(net, [2,2], [2,2], padding='SAME', scope='pool3')
                #   down scale by 2
                net = self._conv_bias_relu(net, 512, 3, 1, 'SAME', 'conv4_1', use_loaded=True, lock=True)
                net = self._conv_bias_relu(net, 512, 3, 1, 'SAME', 'conv4_2', use_loaded=True, lock=True)
        return net

    def _cpm_stage(self, feat_map, stage_num, last_stage = None, load_pretrained=False, lock=False):
        """ CPM stage Sturcture
        Args:
            feat_map    : Input Tensor from feature extractor
            last_stage  : Input Tensor from below
            stage_num   : stage number
            name        : name of the stage
        """
        with tf.variable_scope('CPM_stage'+str(stage_num)):
            if stage_num == 1:
                net = self._conv_bn_relu(feat_map, 256, 3, 1, 'SAME', 'conv4_3_CPM', regularizers=self.regularizers, use_loaded=load_pretrained, lock=lock)
                net = self._conv_bn_relu(net, 256, 3, 1, 'SAME', 'conv4_4_CPM', regularizers=self.regularizers, use_loaded=load_pretrained, lock=lock)
                net = self._conv_bn_relu(net, 256, 3, 1, 'SAME', 'conv4_5_CPM', regularizers=self.regularizers, use_loaded=load_pretrained, lock=lock)
                net = self._conv_bn_relu(net, 256, 3, 1, 'SAME', 'conv4_6_CPM', regularizers=self.regularizers, use_loaded=load_pretrained, lock=lock)
                net = self._conv_bn_relu(net, 128, 3, 1, 'SAME', 'conv4_7_CPM', regularizers=self.regularizers, use_loaded=load_pretrained, lock=lock)
                map = self._conv_bn_relu(net, 512, 1, 1, 'SAME', 'conv5_1_CPM', regularizers=self.regularizers, use_loaded=load_pretrained, lock=lock)
                map = self._conv_bn(map, self.joint_num+1, 1, 1, 'SAME', 'conv5_2_CPM', regularizers=self.regularizers, use_loaded=load_pretrained, lock=lock)
                return map, net
            elif stage_num > 1:
                net = tf.concat([feat_map, last_stage], 3)
                net = self._conv_bn_relu(net, 128, 7, 1, 'SAME', 'Mconv1_stage'+str(stage_num), regularizers=self.regularizers, use_loaded=load_pretrained, lock=lock)
                net = self._conv_bn_relu(net, 128, 7, 1, 'SAME', 'Mconv2_stage'+str(stage_num), regularizers=self.regularizers, use_loaded=load_pretrained, lock=lock)
                net = self._conv_bn_relu(net, 128, 7, 1, 'SAME', 'Mconv3_stage'+str(stage_num), regularizers=self.regularizers, use_loaded=load_pretrained, lock=lock)
                map = self._conv_bn_relu(net, 128, 7, 1, 'SAME', 'Mconv4_stage'+str(stage_num), regularizers=self.regularizers, use_loaded=load_pretrained, lock=lock)
                map = self._conv_bn_relu(map, 128, 7, 1, 'SAME', 'Mconv5_stage'+str(stage_num), regularizers=self.regularizers, use_loaded=load_pretrained, lock=lock)
                map = self._conv_bn_relu(map, 128, 1, 1, 'SAME', 'Mconv6_stage'+str(stage_num), regularizers=self.regularizers, use_loaded=load_pretrained, lock=lock)
                map = self._conv_bn(map, self.joint_num+1, 1, 1, 'SAME', 'Mconv7_stage'+str(stage_num), regularizers=self.regularizers, use_loaded=load_pretrained, lock=lock)
                return map, net
