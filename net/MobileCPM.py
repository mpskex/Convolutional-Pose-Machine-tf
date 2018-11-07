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
"""
import os
import numpy as np
import tensorflow as tf
import CPM


class MobileCPM(CPM.CPM):
    """
    CPM net
    """

    def __init__(self, base_lr=0.0005, in_size=224, out_size=None, batch_size=16, epoch=20, dataset=None, log_dir=None, stage=6,
                 epoch_size=1000, w_summary=True, training=True, joints=None, cpu_only=False, pretrained_model='model/mobilenetv1_1.0.npy',
                 load_pretrained=False, predict=False, name="MobileCPM"):
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
        super(MobileCPM, self).__init__(base_lr, in_size, out_size, batch_size, epoch, dataset, log_dir, stage,
                            epoch_size, w_summary, training, joints, cpu_only, pretrained_model, load_pretrained, predict, name)

    def build_ph(self):
        """ Building Placeholder in tensorflow session
        :return:
        """
        #   Valid & Train input
        #   input image : channel 3
        self.img = tf.placeholder(tf.float32, 
            shape=[None, self.in_size, self.in_size, 3], name="img_in")
        #   input center map : channel 1 (downscale by 8)
        self.weight = tf.placeholder(tf.float32,
            shape=[None, self.joint_num+1])

        #   Train input
        #   input ground truth : channel 1 (downscale by 8)
        self.joint_map_gt = tf.placeholder(tf.float32, 
            shape=[None, self.stage, self.out_size, self.out_size, self.joint_num+1], name="gtmap")

        print("- PLACEHOLDER build finished!")

    def build_train_op(self):
        """ Building training associates: losses & loss summary
        :return:
        """
        #   Optimizer
        with tf.name_scope('loss'):
            loss = tf.multiply(self.weight, tf.reduce_sum(tf.nn.l2_loss(self.joint_map - self.joint_map_gt)))
            self.losses.append(loss)
            self.total_loss = tf.reduce_mean(self.losses)
            self.summ_scalar_list.append(tf.summary.scalar("total loss", self.total_loss))
            self.summ_scalar_list.append(tf.summary.scalar("lr", self.learning_rate))
            print("- LOSS & SCALAR_SUMMARY build finished!")
        with tf.name_scope('optimizer'):
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
                #   Global train
                self.train_step.append(self.optimizer.minimize(self.total_loss/self.batch_size,
                                                                global_step=self.global_step))
        print("- OPTIMIZER build finished!")

    def train(self):
        """ Training Progress in MobileCPM

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
                                                    self.weight:_train_batch[2]})
                #   summaries
                if _iter_count % 10 == 0:
                    _test_batch = next(self.valid_gen)
                    #   doing the scalar summary
                    summ_scalar_out, summ_img_out, summ_acc_out, summ_hist_out, jloss_out = self.sess.run(
                                [self.summ_scalar, self.summ_image, self.summ_accuracy, self.summ_histogram, self.total_loss],
                                                    feed_dict={self.img: _test_batch[0],
                                                                self.joint_map_gt: _test_batch[1],
                                                                self.joint_weight:_test_batch[2]})
                    for n in [summ_scalar_out, summ_img_out, summ_acc_out, summ_hist_out]:
                        self.writer.add_summary(n, _iter_count)
                    print("epoch ", _epoch_count, " iter ", _iter_count, [jloss_out])

                if _iter_count % 500 == 0:
                    #   generate heatmap from the network
                    maps = self.sess.run(self.joint_map,
                            feed_dict={self.img: _test_batch[0],
                                    self.joint_map_gt: _test_batch[1],
                                    self.weight: _test_batch[2]})
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
            #   Note:   This may crash in TF 1.10 (For convenient I just comment this line and use numpy file)
                self.saver.save(self.sess, os.path.join(self.log_dir, "model.ckpt"), n)
    
    def BuildMobileV1Model(self):
        #   input
        with tf.name_scope('input'):
            self.build_ph()
        #   assertion
        assert self.img!=None and self.joint_map_gt!=None
        self.joint_map = self._feature_extractor(self.img, 'MobileNet_V1', 'MobileNetV1')
        self.saver = tf.train.Saver()
        self.writer.add_graph(self.sess.graph)
        print("[*]\tModel Built")

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
        if net_type == 'MobileNet_V1':
            with tf.variable_scope(net_type):
                net = self._conv_bn_relu(inputs, 32, 3, 2, 'SAME', 'Conv2d_0', use_loaded=True, lock=True)
                net = self._separable_conv(net, 64, 3, 1, 'SAME', 'Conv2d_1', use_loaded=True, lock=True)
                net = self._separable_conv(net, 128, 3, 2, 'SAME', 'Conv2d_2', use_loaded=True, lock=True)
                net = self._separable_conv(net, 128, 3, 1, 'SAME', 'Conv2d_3', use_loaded=True, lock=True)
                net = self._separable_conv(net, 256, 3, 2, 'SAME', 'Conv2d_4', use_loaded=True, lock=True)
                net = self._separable_conv(net, 256, 3, 1, 'SAME', 'Conv2d_5', use_loaded=True, lock=True)
                net = self._separable_conv(net, 512, 3, 1, 'SAME', 'Conv2d_6', use_loaded=True, lock=True)
                net = self._separable_conv(net, 512, 3, 1, 'SAME', 'Conv2d_7', use_loaded=True, lock=True)
                net = self._separable_conv(net, 512, 3, 1, 'SAME', 'Conv2d_8', use_loaded=True, lock=True)
        return net

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
            fmap = self._feature_extractor(image, 'MobileNet_V1', 'MobileNetV1')    
            self.summ_image_list.append(tf.summary.image("MobileNet", tf.expand_dims(tf.expand_dims(tf.reduce_sum(fmap, axis=[-1])[0],-1),0), max_outputs=1))
            stage = [None] * self.stage
            stage[0] = self._cpm_stage(fmap, 1, None)
            for t in range(2,self.stage+1):
                stage[t-1] = self._cpm_stage(fmap, t, stage[t-2])
            #   RETURN SIZE:
            #       batch_size * stage_num * in_size/8 * in_size/8 * joint_num
            return tf.nn.sigmoid(tf.stack(stage, axis=1 , name= 'stack_output'),name = 'final_output')

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
                net = self._separable_conv(feat_map, 256, 3, 1, 'SAME', 'conv4_3_CPM', use_loaded=self.load_pretrained, lock=not self.training)
                net = self._separable_conv(net, 256, 3, 1, 'SAME', 'conv4_4_CPM', use_loaded=self.load_pretrained, lock=not self.training)
                net = self._separable_conv(net, 256, 3, 1, 'SAME', 'conv4_5_CPM', use_loaded=self.load_pretrained, lock=not self.training)
                net = self._separable_conv(net, 256, 3, 1, 'SAME', 'conv4_6_CPM', use_loaded=self.load_pretrained, lock=not self.training)
                net = self._separable_conv(net, 128, 3, 1, 'SAME', 'conv4_7_CPM', use_loaded=self.load_pretrained, lock=not self.training)
                net = self._conv_bn_relu(net, 512, 1, 1, 'SAME', 'conv5_1_CPM', use_loaded=self.load_pretrained, lock=not self.training)
                net = self._conv_bn(net, self.joint_num+1, 1, 1, 'SAME', 'conv5_2_CPM', use_loaded=self.load_pretrained, lock=not self.training)
                return net
            elif stage_num > 1:
                net = tf.concat([feat_map, last_stage], 3)
                net = self._separable_conv(net, 128, 7, 1, 'SAME', 'Mconv1_stage'+str(stage_num), use_loaded=self.load_pretrained, lock=not self.training)
                net = self._separable_conv(net, 128, 7, 1, 'SAME', 'Mconv2_stage'+str(stage_num), use_loaded=self.load_pretrained, lock=not self.training)
                net = self._separable_conv(net, 128, 7, 1, 'SAME', 'Mconv3_stage'+str(stage_num), use_loaded=self.load_pretrained, lock=not self.training)
                net = self._separable_conv(net, 128, 7, 1, 'SAME', 'Mconv4_stage'+str(stage_num), use_loaded=self.load_pretrained, lock=not self.training)
                net = self._separable_conv(net, 128, 7, 1, 'SAME', 'Mconv5_stage'+str(stage_num), use_loaded=self.load_pretrained, lock=not self.training)
                net = self._conv_bn_relu(net, 128, 1, 1, 'SAME', 'Mconv6_stage'+str(stage_num), use_loaded=self.load_pretrained, lock=not self.training)
                net = self._conv_bn(net, self.joint_num+1, 1, 1, 'SAME', 'Mconv7_stage'+str(stage_num), use_loaded=self.load_pretrained, lock=not self.training)
                return net
