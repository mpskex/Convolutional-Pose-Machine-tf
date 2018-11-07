# coding: utf-8
"""
    Attentional Multi-person Pose Estimation
        For Single Person Pose Estimation
    Human Pose Estimation Project in Lab of IP
    Author: Liu Fangrui aka mpsk
        Beijing University of Technology
            College of Computer Science & Technology
    Experimental Code
        !!DO NOT USE IT AS DEPLOYMENT!!

    This is a experiment code in imageLab in BJUT
    Liu et.al
"""
import os
import time
import numpy as np
import tensorflow as tf
import CPM


class PPNnMask(CPM.CPM):
    """
    Attentional Multi-person Pose
    """

    def __init__(self, base_lr=0.0005, in_size=368, out_size=None, batch_size=16, epoch=20, dataset=None, anchors=None, rois_max=16, log_dir=None,
                 stage=6, epoch_size=1000, w_summary=True, training=True, 
                 train_joint_net=True,
                 train_insmask_net=True,
                 joints=None, cpu_only=False, pretrained_model='vgg19.npy', load_pretrained=False,
                 predict=False, name='model/PPNnMask', _type='V4'):
        super(PPNnMask, self).__init__(base_lr, in_size, out_size, batch_size, epoch, dataset, log_dir, stage,
                                   epoch_size, w_summary, training, joints, cpu_only, pretrained_model, load_pretrained, predict, name)
        self.anchors = anchors
        self.ppn_lambda = 10
        self.rois_max = rois_max
        if training == True:
            self.nms_score_threshold = float('-inf')
            self.nms_iou_threshold = 0.2
        else:
            self.nms_score_threshold = 0.98
            self.nms_iou_threshold = 0.2
        self.train_joint_net = train_joint_net
        self.train_insmask_net = train_insmask_net

        self.version = int(_type.split('V')[-1])
        if self.version == 5 or self.version == 8:
            self.boxmap_out_size = self.out_size / 2
            print "[!]\tSetting down-scaling from 8x to 16x!"
        else:
            self.boxmap_out_size = self.out_size


    def build_ph(self):
        """ Building Placeholder in tensorflow session
        :return:
        """
        super(PPNnMask, self).build_ph()
        self.ppn_labels = tf.placeholder(
            tf.int32, shape=[None, self.boxmap_out_size, self.boxmap_out_size, len(self.anchors)])
        self.ppn_bbox_targets = tf.placeholder(
            tf.float32, shape=[None, self.boxmap_out_size, self.boxmap_out_size, 4*len(self.anchors)])
        self.mask_chw_gt = tf.placeholder(
            tf.float32, shape=[None, self.out_size, self.out_size, self.rois_max], name="channel_wise_segmap")
        #
        with tf.variable_scope("Position_Map"):
            #   Build Position Map for offset

            #   for x (out_size, 1)
            #   for y (1, out_size)
            x = np.expand_dims(np.array(range(self.boxmap_out_size)), axis=-1)
            y = np.expand_dims(np.array(range(self.boxmap_out_size)), axis=0)
            x = np.tile(x, (1, self.boxmap_out_size))
            y = np.tile(y, (self.boxmap_out_size, 1))
            #   self.pos_map has shape of (out_size, out_size, 1, 2)
            x_y = np.expand_dims(np.stack([x, y], axis=-1), axis=2)
            #   (out_size, out_size, anchors, 2)
            x_y = np.tile(x_y, (1, 1, len(self.anchors), 1))

            #   (anchors, 2)
            np_anchor = np.array(self.anchors)
            #   (1, 1, anchors, 2)
            w_h = np.expand_dims(np.expand_dims(np_anchor, axis=0), axis=0)
            #   (out_size, out_size, anchors, 2)
            w_h = np.tile(w_h, (self.boxmap_out_size, self.boxmap_out_size, 1, 1))

            #   (out_size, out_size, anchors, 4)
            print x_y.shape
            print w_h.shape
            self.pos_map_single = np.concatenate([x_y, w_h], axis=-1)

            #   (out_size, out_size, anchors, 4)
            self.pos_map_single = np.expand_dims(self.pos_map_single, axis=0)
            #   get pos_map shape
            print self.pos_map_single[0, 1, 2, 0]

            self.pos_map_single = tf.constant(self.pos_map_single, dtype=tf.float32)

    def build_monitor(self):
        """ Building image summaries

        :return:
        """
        with tf.device(self.cpu):
            #   calculate the return full map
            self.summ_image_list.append(tf.summary.image("image", tf.expand_dims(self.img[0], 0), max_outputs=3))
            for n in range(len(self.anchors)):
                self.summ_image_list.append(
                    tf.summary.image("score_map_anchor_"+str(self.anchors[n][0])+'*'+str(self.anchors[n][1]), 
                            tf.expand_dims(tf.expand_dims(self.ppn_cls_map_softmax[0,:,:,n,1], axis=-1), axis=0)))
            for n in range(len(self.anchors)):
                self.summ_image_list.append(
                    tf.summary.image("GT_anchor_"+str(self.anchors[n][0])+'*'+str(self.anchors[n][1]), 
                            tf.cast(tf.expand_dims(tf.expand_dims(self.ppn_labels[0,:,:,n], axis=-1), axis=0), tf.float32)))
            if not self.no_mask_net:
                self.summ_image_list.append(tf.summary.image("image", tf.expand_dims(self.mask_mono_pred[0, :, :, 1], 0), max_outputs=3))
            print "\t* monitor image have shape of ", tf.expand_dims(self.img[0], 0).shape

            print "- IMAGE_SUMMARY build finished!"

    def build_train_op(self):
        """ Building training associates: losses & loss summary

        TODO:   
            Make declaration for the dimensional information
        """
        #   Optimizer
        with tf.name_scope('loss'):
            '''
            with tf.variable_scope("JointLoss"):
                #   Keypoint Branch Loss
                #       *   Multi stages summed up loss
                self.joint_total_loss = tf.multiply(self.joint_weight, tf.reduce_sum(
                    tf.nn.l2_loss(self.joint_map - self.joint_map_gt)))
                self.joint_total_loss = tf.reduce_sum(
                    tf.nn.l2_loss(self.joint_map - self.joint_map_gt))
            '''
            sample_size = 32
            #   DOING:
            #   Person Proposal Net
            #   *   Anchor regression loss
            # input shape dimensions
            with tf.variable_scope("PPNLoss"):
                # INPUT:
                # Stack all classification scores into 2D matrix
                self.ppn_cls_score = tf.reshape(self.ppn_cls_map, [-1, 2])
                # Stack labels
                # extend label to one_hot vairable
                self.ex_ppn_labels = tf.reshape(self.ppn_labels, [-1])
                # Ignore positions whose label=-1 (Neither object nor background: IoU between 0.3 and 0.7)
                # Reshape to [-1,2] to calculate cross encropy loss
                self.ppn_cls_score = tf.reshape(tf.gather(
                    self.ppn_cls_score, tf.where(tf.not_equal(self.ex_ppn_labels, -1))), [-1, 2])
                # Keep only non-(-1) value whose IoU > 0.7
                self.ex_ppn_labels = tf.reshape(
                    tf.gather(self.ex_ppn_labels, tf.where(tf.not_equal(self.ex_ppn_labels, -1))), [-1])

                #   According to the original Faster R-CNN Training strategy
                #   we ramdomly choose positive and negative samples with pos:neg = 1:1
                #   and pad the empty positive with negative samples
                self.ppn_score_label = tf.concat([self.ppn_cls_score, tf.cast(tf.expand_dims(self.ex_ppn_labels, axis=-1), tf.float32)], -1)
                self.ex_ppn_cls_pos = tf.gather(self.ppn_score_label, tf.where(tf.equal(self.ex_ppn_labels, 1)))
                self.ex_ppn_cls_neg = tf.gather(self.ppn_score_label, tf.where(tf.equal(self.ex_ppn_labels, 0)))
                self.ex_ppn_cls_pos, self.ex_ppn_cls_neg = self.ex_ppn_cls_pos[:,0,:], self.ex_ppn_cls_neg[:,0,:]
                print self.ex_ppn_cls_pos.shape
                #   create index
                self.ex_ppn_cls_pos_idx = tf.random_shuffle(tf.range(tf.shape(self.ex_ppn_cls_pos)[0]))
                self.ex_ppn_cls_neg_idx = tf.random_shuffle(tf.range(tf.shape(self.ex_ppn_cls_neg)[0]))
                #   if there are not enough positive then fill it with negative
                self.ex_ppn_cls_pos = tf.gather(self.ex_ppn_cls_pos, self.ex_ppn_cls_pos_idx)
                self.ex_ppn_cls_neg = tf.gather(self.ex_ppn_cls_neg, self.ex_ppn_cls_neg_idx)
                self.ex_ppn_cls_pos = tf.cond(tf.shape(self.ex_ppn_cls_pos)[0] > sample_size, 
                                                lambda: self.ex_ppn_cls_pos[:sample_size],
                                                lambda: tf.concat([self.ex_ppn_cls_pos, 
                                                                    self.ex_ppn_cls_neg[tf.shape(self.ex_ppn_cls_pos)[0]-sample_size:]], 0))
                self.ex_ppn_cls_neg = self.ex_ppn_cls_neg[:sample_size]
                self.ppn_score_label = tf.concat([self.ex_ppn_cls_pos, self.ex_ppn_cls_neg], 0)
                self.ppn_cls_score = self.ppn_score_label[:, :2]
                self.ex_ppn_labels = tf.cast(self.ppn_score_label[:, -1], tf.int32)
                # '''

                # Cross entropy error
                self.ppn_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.ppn_cls_score, labels=self.ex_ppn_labels))

                # INPUT:
                self.ppn_labels_abs = tf.abs(self.ppn_labels)
                self.ppn_labels_abs = tf.tile(self.ppn_labels_abs, [1, 1, 1, 4])
                print "[*]\tself.ppn_labels has shape of ", self.ppn_labels.shape
                # How far off was the prediction?
                # ( pred - label ) * ppn_inside_weights. It means to cal bbox loss only for positive anchor
                # Smooth_L1 result
                self.ppn_bbox_reg = tf.losses.huber_loss(
                    self.ppn_bbox_targets, self.ppn_bbox_pred, weights=self.ppn_labels_abs, delta=3.0)
                # Mul Loss Constant lambda for weighting bounding box loss with classification loss
                self.ppn_bbox_reg = self.ppn_lambda * self.ppn_bbox_reg

                #   temporalily stop training offsets
                #self.ppn_total_loss = self.ppn_cross_entropy + self.ppn_bbox_reg
                self.ppn_total_loss = self.ppn_cross_entropy

            if not self.no_mask_net:
                with tf.variable_scope("Mask_Loss"):
                    #   TODO:   pick the matching IoU > 0.5 RoI
                    #           Think of the failing cases... 
                    #           If we generate the masks then fusion them together...
                    #           like mix them with bounding box IoUs

                    #   This will pick the max (avoiding overflow)
                    self.mask_mono_gt = tf.reduce_max(self.mask_chw_gt, axis=-1, keepdims=True)
                    self.mask_mono_pred = tf.stack([self.dw_mask[:, :, :, -1],
                                                    tf.reduce_max(self.dw_mask[:, :, :, :self.rois_max], axis=-1),
                                                    ])

                    #   for the ground truth is just label
                    self.mask_mono_gt = tf.reshape(self.mask_mono_gt, [-1])
                    self.masks_pred = tf.reshape(self.mask_mono_pred, [-1, 2])
                    #   cross_entropy mask loss
                    self.mask_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                                    logits=self.masks_pred, labels=tf.cast(self.mask_mono_gt, tf.int32)))

                with tf.variable_scope("Mask_RCNN_Loss"):
                    self.mrcnn_loss = self.mask_loss + self.ppn_total_loss

            # '''
            with tf.variable_scope("summaries"):
                if not self.no_mask_net:
                    self.summ_scalar_list.append(
                        tf.summary.scalar("Mask Loss", self.mask_loss))
                    self.summ_scalar_list.append(
                        tf.summary.scalar("Mask-RCNN Total Loss", self.mrcnn_loss))
                self.summ_scalar_list.append(
                    tf.summary.scalar("PPN Offset Loss", self.ppn_bbox_reg))
                self.summ_scalar_list.append(
                    tf.summary.scalar("PPN Class Loss", self.ppn_cross_entropy))
                self.summ_scalar_list.append(
                        tf.summary.scalar("PPN Total Loss", self.ppn_total_loss))
                # '''
                self.summ_scalar_list.append(
                    tf.summary.scalar("lr", self.learning_rate))
                print("- LOSS & SCALAR_SUMMARY build finished!")
        with tf.name_scope('optimizer'):
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):

                #   2e-2 for  PPNnet
                self.ppnet_optimizer = tf.train.GradientDescentOptimizer(
                    50*self.learning_rate)
                self.masknet_optimizer = tf.train.GradientDescentOptimizer(
                    30*self.learning_rate)

                #   mask RCNN + PPN
                if self.train_insmask_net:
                    self.train_step.append(self.ppnet_optimizer.minimize(self.ppn_total_loss,
                                                                        global_step=self.global_step))
                    if not self.no_mask_net:
                        self.train_step.append(self.masknet_optimizer.minimize(self.mask_loss,
                                                                        global_step=self.global_step))
                    

        print("- OPTIMIZER build finished!")

    def BuildModel(self, debug=False, no_mask_net=False, lock_first_stage=False):
        """ Building model in tensorflow session

        :return:
        """
        tf.reset_default_graph()
        self.sess = tf.Session()
        if self.training:
            self.writer = tf.summary.FileWriter(self.log_dir)
            self.global_step = tf.Variable(0, trainable=False)
            self.learning_rate = tf.train.exponential_decay(self.base_lr,
                self.global_step, 50*self.epoch_size, 0.333,
                staircase=True)
        #   input
        self.no_mask_net = no_mask_net
        with tf.name_scope('input'):
            self.build_ph()
            print("- PLACEHOLDER build finished!")
        #   assertion
        assert self.img != None and self.joint_map_gt != None

        #   Build net structure
        with tf.variable_scope("PPNnMask"):
            #   individual branch
            self.fmap = self._feature_extractor(
                self.img, 'VGG', 'Feature_Extractor')
            self.joint_map = [None] * self.stage
            with tf.variable_scope("Extract_Stage"):
                self.ppn_cls_map, \
                self.ppn_bbox_pred = self.PPNet(self.fmap,
                                                len(self.anchors),
                                                load_pretrained=self.load_pretrained,
                                                lock=not self.training)
                self.ppn_cls_map = tf.reshape(self.ppn_cls_map, (-1, self.boxmap_out_size, self.boxmap_out_size, len(self.anchors), 2))
                self.ppn_cls_map_softmax = tf.nn.softmax(self.ppn_cls_map)
                self.ppn_result = self.proposal_layer(
                                            self.ppn_cls_map_softmax,
                                            self.ppn_bbox_pred,
                                            self.pos_map_single,
                                            self.anchors,
                                            self.boxmap_out_size,
                                            nms_iou_threshold=self.nms_iou_threshold,
                                            nms_score_threshold=self.nms_score_threshold,
                                            pre_nms_resrv=1000,
                                            post_nms_resrv=self.rois_max)
                with tf.variable_scope("Reshaping"):
                    #   roi_feat should have shape (person, batch_size, out_size, out_size, channel)
                    self.ppn_bbox = self.ppn_result[:, :4] * (self.out_size / self.boxmap_out_size)
                    self.ppn_result = tf.concat([self.ppn_bbox, tf.expand_dims(self.ppn_result[:,-1], axis=-1)], 1)
                    self.ppn_batch_ind = tf.cast(self.ppn_result[:, -1], tf.int32)
                #   feed only stage-1 feature into the roi
                self.roi_feat = self.roi_align(self.fmap,
                                            self.ppn_bbox,
                                            self.ppn_batch_ind,
                                            self.out_size,
                                            512)
                
                if not self.no_mask_net:
                    self.masks = self.mask_net(self.roi_feat,
                                                load_pretrained=self.load_pretrained,
                                                lock=not self.training)
                    
                    self.dw_mask, self.rois_mask_rsz = self.dispatch_layer(self.ppn_bbox,
                                                        self.masks,
                                                        self.ppn_batch_ind,
                                                        self.batch_size,
                                                        self.rois_max,
                                                        self.out_size,
                                                        self.out_size,
                                                        2)
                    self.dw_mask = tf.concat([self.dw_mask[:,:,:,:,1], tf.reduce_min(self.dw_mask[:,:,:,:,0], axis=-1, keepdims=True)], 3)
                    print "[*]\tDepth-wise mask has shape of ", self.dw_mask.shape

        if not debug:
            #   the net
            if self.training:
                #   train op
                with tf.name_scope('train'):
                    self.build_train_op()
                with tf.name_scope('image_summary'):
                    self.build_monitor()
            #   initialize all variables
            self.sess.run(tf.global_variables_initializer())
            if self.training:
                self.saver = tf.train.Saver()
                #   merge all summary
                self.summ_image = tf.summary.merge(self.summ_image_list)
                self.summ_scalar = tf.summary.merge(self.summ_scalar_list)
                self.writer.add_graph(self.sess.graph)
        print("[*]\tModel Built")

    def PPNet(self, feat_map, anchors_per_location, load_pretrained=False, lock=False, name="PPNet"):
        """ Person Proposal Net
        """
        with tf.variable_scope(name):
            #   shared conv in Person Proposal Net
            with tf.variable_scope('PPNet_Conv'):
                # ''' >>>> VGG Feature
                #   This version is digest features from VGG
                if self.version == 5:
                    net = tf.contrib.layers.max_pool2d(feat_map, [2,2], [2,2], padding='SAME', scope='pool4')
                    net = self._conv_bias_relu(net, 512, 3, 1, 'SAME', 'conv5_1', use_loaded=True, lock=lock)
                    print "[!]\tloaded conv5_1 for V5 model!"
                elif self.version == 8:
                    net = tf.contrib.layers.max_pool2d(feat_map, [2,2], [2,2], padding='SAME', scope='pool4')
                    net = self._conv_bias_relu(net, 512, 3, 1, 'SAME', 'conv5_1', use_loaded=True, lock=lock)
                    net = self._conv_bias_relu(net, 512, 3, 1, 'SAME', 'conv5_2', use_loaded=True, lock=lock)
                    net = self._conv_bias_relu(net, 512, 3, 1, 'SAME', 'conv5_3', use_loaded=True, lock=lock)
                    print "[!]\tloaded conv5_1 conv5_2 conv5_3 for V8 model!"
                elif self.version !=5 and self.version != 8:
                    net = self._conv_bias_relu(feat_map, 512, 3, 1, 'SAME', 'conv5_1', use_loaded=load_pretrained, lock=lock)
                    net = self._conv_bn_relu(net, 512, 3, 1, 'SAME', 'conv5_2', use_loaded=load_pretrained, lock=lock)
                    net = self._conv_bn_relu(net, 512, 3, 1, 'SAME', 'conv5_3', use_loaded=load_pretrained, lock=lock)
                shared = self._conv_bn_relu(net, 512, 3, 1, 'SAME', 'ppn_conv_shared', use_loaded=load_pretrained, lock=lock)

                #   Anchor score (FG / BG) [batch_size, width, height, anchors_per_location * 2]
                score_map = self._conv(shared, 2*anchors_per_location, 1, 1, 'SAME', name='ppn_score_map', use_loaded=load_pretrained, lock=lock)
                #   Anchor location prediction
                bbox_map = self._conv(shared, 4*anchors_per_location, 1, 1, 'SAME', name='ppn_bbox_map', use_loaded=load_pretrained, lock=lock)
        return score_map, bbox_map

    def CPMNet(self, fmap, name='CPM', lock_first_stage=False):
        """ CPM Net Structure
        Args:
            fmap           : Featrue Map from VGG-19 / ResNet 50
        Return:
            stacked heatmap : Heatmap NSHWC format
                                size:   batch_size * stage_num * in_size/8 * in_size/8 * joint_num
        """
        with tf.variable_scope(name):
            stage = [None] * self.stage
            joint_feat = [None] * self.stage
            stage[0], joint_feat[0] = self._cpm_stage(fmap, 1, None, load_pretrained=lock_first_stage, lock=lock_first_stage)
            for t in range(2, self.stage+1):
                stage[t-1], joint_feat[t - 1] = self._cpm_stage(fmap, t, stage[t-2])
            #   RETURN SIZE:
            #       batch_size * stage_num * in_size/8 * in_size/8 * joint_num
            return tf.nn.sigmoid(tf.stack(stage, axis=1, name='stack_output'), name='final_output'), joint_feat

    def mask_net(self, roi, load_pretrained=False, lock=False, name="MaskNet"):
        """ Mask Sub Net
        Args:
            roi         :   Input Tensor that recieve RoIs
                            NOTE:   the smallest mini-batch regression unit is roi
            name        :   name of the stage
        """
        with tf.variable_scope(name):
            # ROI Pooling
            # Shape: [batch, pool_height, pool_width, channels]
            net = self._conv_bn_relu(
                roi, 256, 3, 1, 'SAME', 'mrcnn_mask_conv1', use_loaded=load_pretrained, lock=lock)
            net = self._conv_bn_relu(
                net, 256, 3, 1, 'SAME', 'mrcnn_mask_conv2', use_loaded=load_pretrained, lock=lock)
            net = self._conv_bn_relu(
                net, 256, 3, 1, 'SAME', 'mrcnn_mask_conv3', use_loaded=load_pretrained, lock=lock)
            net = self._conv_bn_relu(
                net, 256, 3, 1, 'SAME', 'mrcnn_mask_conv4', use_loaded=load_pretrained, lock=lock)
            _map = self._conv_bn(net, 2, 1, 1, 'SAME', 'mrcnn_mask_conv5',
                                use_loaded=load_pretrained, lock=lock)
        return _map

    def train(self):
        """ Training Progress 
        """
        _epoch_count = 0
        _iter_count = 0

        #   datagen from Hourglass
        self.generator = self.dataset.generator(
            self.batch_size, stacks=self.stage, sample_set='train', box_downscale=(self.out_size/self.boxmap_out_size))
        self.valid_gen = self.dataset.generator(
            self.batch_size, stacks=self.stage, sample_set='val', box_downscale=(self.out_size/self.boxmap_out_size))

        for n in range(self.epoch):
            for m in range(self.epoch_size):
                #   datagen from hourglass
                #   running training steps
                #   TODO: place holder initialization
                train_img, train_joint_map_gt, _, train_rmask, train_bbox_offset, train_score = next(self.generator)
                self.sess.run(self.train_step, feed_dict={self.img: train_img,
                                                          self.joint_map_gt: train_joint_map_gt,
                                                          self.ppn_bbox_targets: train_bbox_offset,
                                                          self.ppn_labels: train_score,
                                                          self.mask_chw_gt: train_rmask,
                                                          })
                print("iter:", _iter_count)
                #   summaries
                if _iter_count % 10 == 0:
                    test_img, test_joint_map_gt, _, test_rmask, test_bbox_offset, test_score = next(self.valid_gen)
                    
                    #   doing the scalar summary
                    summ_scalar_out,\
                    summ_img_out,\
                    ppn_total_loss = self.sess.run(
                                [self.summ_scalar, self.summ_image, self.ppn_total_loss],
                                                feed_dict={self.img: test_img,
                                                          self.joint_map_gt: test_joint_map_gt,
                                                          self.ppn_bbox_targets: test_bbox_offset,
                                                          self.ppn_labels: test_score,
                                                          self.mask_chw_gt: test_rmask,
                                                          #   add a batch feeder
                                                          })
                    for n in [summ_scalar_out,
                                summ_img_out]:
                        self.writer.add_summary(n, _iter_count)
                    print("epoch ", _epoch_count, " iter ", _iter_count, [ppn_total_loss])
                _iter_count += 1
                self.writer.flush()
            #   doing save numpy params
            self.save_npy()
            _epoch_count += 1
            #   save model every epoch
            if self.log_dir is not None:
                #   Note:   This may crash in TF 1.10 (For convenient I just comment this line and use numpy file)
                self.saver.save(self.sess, os.path.join(self.log_dir, "model.ckpt"), n)
                pass
