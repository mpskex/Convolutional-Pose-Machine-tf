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
            This layer library contains Faster R-CNN component
            This is re-implementation all by my self, and tested
            with more than thousands of samples.
            P.S.: The test file can be found in unit-test/
"""

'''
from nms.gpu_nms import gpu_nms
from nms.cpu_nms import cpu_nms
'''
import numpy as np
import tensorflow as tf
""" 
Convolution with batch norm
    Convolution weight              :           weights             0
    Batch Normalization Beta        :           beta(offset)        1
    Batch Normalization Gamma       :           gamma(scale)        2
    Batch Normalization Moving Mean :           movmean             3
    Batch Normalization Mov Var     :           movvar              4

Convolution with bias
    Convolution weight              :           weights             0
    Convolution bias                :           bias                1
"""


#   ======= Net Component ========
class LayerLibrary(object):
    def _batch_norm(self, inputs, decay=0.99, epsilon=1e-3, use_loaded=False, lock=False, is_training=False, name='BatchNorm'):
        """ BatchNormalization Layers
        """
        with tf.variable_scope(name):
            if self.pretrained_model is None:
                use_loaded = False
                print("[!]\tWarning:\tPretrained model not loaded...Using initial value! name: ", name)
            if use_loaded:
                if not self.training:
                    beta = tf.constant(self.pretrained_model[name][1], name='bn_beta')
                    gamma = tf.constant(self.pretrained_model[name][2], name='bn_gamma')
                    mean = tf.constant(self.pretrained_model[name][3], name='bn_moving_mean')
                    variance = tf.constant(self.pretrained_model[name][4], name='bn_moving_var')
                    print("[!]\tLayer's BN Param restored! name of ", name)
                else:
                    beta = tf.Variable(self.pretrained_model[name][1], name='bn_beta', trainable=not lock)
                    gamma = tf.Variable(self.pretrained_model[name][2], name='bn_gamma', trainable=not lock)
                    mean = tf.Variable(self.pretrained_model[name][3], name='bn_moving_mean', trainable=False)
                    variance = tf.Variable(self.pretrained_model[name][4], name='bn_moving_var', trainable=False)
                    if lock:
                        print("[!]\tLocked ", name + "/BatchNorm", " parameters")
            else:
                gamma = tf.Variable(tf.ones([inputs.get_shape().as_list()[-1]]), name='bn_beta')
                beta = tf.Variable(tf.zeros([inputs.get_shape().as_list()[-1]]), name='bn_gamma')
                mean = tf.Variable(tf.zeros([inputs.get_shape().as_list()[-1]]), name='bn_moving_mean', trainable=False)
                variance = tf.Variable(tf.ones([inputs.get_shape().as_list()[-1]]), name='bn_moving_var', trainable=False)

            if is_training:
                self.var_dict[(name, 1)] = beta
                self.var_dict[(name, 2)] = gamma
                self.var_dict[(name, 3)] = mean
                self.var_dict[(name, 4)] = variance
                batch_mean, batch_variance = tf.nn.moments(inputs, [0, 1, 2], keep_dims=False)

                train_mean = tf.assign(mean, mean * decay + batch_mean * (1 - decay))
                train_variance = tf.assign(variance, variance * decay + batch_variance * (1 - decay))
                with tf.control_dependencies([train_mean, train_variance]):
                    return tf.nn.batch_normalization(inputs, batch_mean, batch_variance, beta, gamma, epsilon)
            else:
                return tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, epsilon)
                
    def _conv(self, inputs, filters, kernel_size=1, strides=1, pad='VALID', name='conv', regularizers=[], use_loaded=False, lock=False):
        """ Spatial Convolution (CONV2D)
        Args:
            inputs			: Input Tensor (Data Type : NHWC)
            filters		    : Number of filters (channels)
            kernel_size	    : Size of kernel
            strides		    : Stride
            pad				: Padding Type (VALID/SAME) # DO NOT USE 'SAME' NETWORK BUILT FOR VALID
            name			: Name of the block
            use_loaded      : Use related name to find weight and bias
            lock            : Lock the layer so the parameter won't be optimized
        Returns:
            conv			: Output Tensor (Convolved Input)
        """
        with tf.variable_scope(name):
            if self.pretrained_model is None:
                use_loaded = False
                print("[!]\tWarning:\tPretrained model not loaded...Using initial value! name: ", name)
            if use_loaded:
                if not self.training:
                    #   TODO:   Assertion
                    kernel = tf.constant(self.pretrained_model[name][0], name='weights')
                    bias = tf.constant(self.pretrained_model[name][1], name='bias')
                    print("[!]\tLayer restored! name of ", name)
                else:
                    kernel = tf.Variable(self.pretrained_model[name][0], name='weights', trainable=not lock)
                    bias = tf.Variable(self.pretrained_model[name][1], name='bias', trainable=not lock)
                    if lock:
                        print("[!]\tLocked ", name, " parameters")
            else:
                # Kernel for convolution, Xavier Initialisation
                kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([kernel_size, kernel_size, inputs.get_shape().as_list()[3], filters]), name='weights')
                bias = tf.Variable(tf.zeros([filters]), name='bias')

            #   save kernel and bias
            self.var_dict[(name, 0)] = kernel
            self.var_dict[(name, 1)] = bias

            #   Collect Regularization Loss
            if lock is False and regularizers is not None:
                for regularizer in regularizers:
                    regularizer.collect(kernel)

            conv = tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1], padding=pad, data_format='NHWC')
            conv_bias = tf.nn.bias_add(conv, bias)
            relu = tf.nn.relu(conv_bias)
            if self.w_summary:
                self.summ_histogram_list.append(tf.summary.histogram(name + 'weights', kernel, collections=['weight']))
                self.summ_histogram_list.append(tf.summary.histogram(name + 'bias', bias, collections=['bias']))
            return relu

    def _conv_bn(self, inputs, filters, kernel_size=1, strides=1, pad='VALID', name='conv', regularizers=[], use_loaded=False, lock=False):
        """ Spatial Convolution (CONV2D) + BatchNormalization
        Args:
            inputs			: Input Tensor (Data Type : NHWC)
            filters		    : Number of filters (channels)
            kernel_size	    : Size of kernel
            strides		    : Stride
            pad				: Padding Type (VALID/SAME) # DO NOT USE 'SAME' NETWORK BUILT FOR VALID
            name			: Name of the block
            use_loaded      : Use related name to find weight and bias
            lock            : Lock the layer so the parameter won't be optimized
        Returns:
            conv			: Output Tensor (Convolved Input)
        """
        with tf.variable_scope(name):
            if self.pretrained_model is None:
                use_loaded = False
                print("[!]\tWarning:\tPretrained model not loaded...Using initial value! name: ", name)
            if use_loaded:
                if not self.training:
                    #   TODO:   Assertion
                    kernel = tf.constant(self.pretrained_model[name][0], name='weights')
                    print("[!]\tLayer restored! name of ", name)
                else:
                    kernel = tf.Variable(self.pretrained_model[name][0], name='weights', trainable=not lock)
                    if lock:
                        print("[!]\tLocked ", name, " parameters")
            else:
                kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([kernel_size, kernel_size, inputs.get_shape().as_list()[3], filters]), name='weights')

            #   save kernel and bias
            self.var_dict[(name, 0)] = kernel

            #   Collect Regularization Loss
            if lock is False and regularizers is not None:
                for regularizer in regularizers:
                    regularizer.collect(kernel)

            conv = tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1], padding=pad, data_format='NHWC')
            norm = self._batch_norm(conv, decay=0.9, epsilon=1e-3, use_loaded=use_loaded, lock=lock, name=name, is_training=self.training)
            if self.w_summary:
                self.summ_histogram_list.append(tf.summary.histogram(name + 'weights', kernel, collections=['weight']))
            return norm

    def _conv_bias_relu(self, inputs, filters, kernel_size=1, strides=1, pad='VALID', name='conv_bias_relu', regularizers=[], use_loaded=False, lock=False):
        """ Spatial Convolution (CONV2D) + Bias + ReLU Activation
        Args:
            inputs			: Input Tensor (Data Type : NHWC)
            filters		    : Number of filters (channels)
            kernel_size	    : Size of kernel
            strides		    : Stride
            pad				: Padding Type (VALID/SAME) # DO NOT USE 'SAME' NETWORK BUILT FOR VALID
            name			: Name of the block
            use_loaded      : Use related name to find weight and bias
            lock            : Lock the layer so the parameter won't be optimized
        Returns:
            norm			: Output Tensor
        """
        with tf.variable_scope(name):
            if self.pretrained_model is None:
                use_loaded = False
                print("[!]\tWarning:\tPretrained model not loaded...Using initial value! name: ", name)
            if use_loaded:
                if not self.training:
                    #   TODO:   Assertion
                    kernel = tf.constant(self.pretrained_model[name][0], name='weights')
                    bias = tf.constant(self.pretrained_model[name][1], name='bias')
                    print("[!]\tLayer restored! name of ", name)
                else:
                    kernel = tf.Variable(self.pretrained_model[name][0], name='weights', trainable=not lock)
                    bias = tf.Variable(self.pretrained_model[name][1], name='bias', trainable=not lock)
                    if lock:
                        print("[!]\tLocked ", name, " parameters")
            else:
                kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([kernel_size, kernel_size, inputs.get_shape().as_list()[3], filters]), name='weights')
                bias = tf.Variable(tf.zeros([filters]), name='bias')

            #   save kernel and bias
            self.var_dict[(name, 0)] = kernel
            self.var_dict[(name, 1)] = bias
            
            #   Collect Regularization Loss
            if lock is False and regularizers is not None:
                for regularizer in regularizers:
                    regularizer.collect(kernel)

            conv = tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1], padding=pad, data_format='NHWC')
            conv_bias = tf.nn.bias_add(conv, bias)
            relu = tf.nn.relu(conv_bias)
            if self.w_summary:
                self.summ_histogram_list.append(tf.summary.histogram(name + 'weights', kernel, collections=['weight']))
                self.summ_histogram_list.append(tf.summary.histogram(name + 'bias', bias, collections=['bias']))
            return relu

    def _conv_bn_relu(self, inputs, filters, kernel_size=1, strides=1, pad='VALID', name='conv_bn_relu', regularizers=[], use_loaded=False, lock=False):
        """ Spatial Convolution (CONV2D) + BatchNormalization + ReLU Activation
        Args:
            inputs			: Input Tensor (Data Type : NHWC)
            filters		    : Number of filters (channels)
            kernel_size	    : Size of kernel
            strides		    : Stride
            pad				: Padding Type (VALID/SAME) # DO NOT USE 'SAME' NETWORK BUILT FOR VALID
            name			: Name of the block
            use_loaded      : Use related name to find weight and bias
            lock            : Lock the layer so the parameter won't be optimized
        Returns:
            norm			: Output Tensor
        """
        with tf.variable_scope(name):
            if self.pretrained_model is None:
                use_loaded = False
                print("[!]\tWarning:\tPretrained model not loaded...Using initial value! name: ", name)
            if use_loaded:
                if not self.training:
                    #   TODO:   Assertion
                    kernel = tf.constant(self.pretrained_model[name][0], name='weights')
                    print("[!]\tLayer restored! name of ", name)
                else:
                    kernel = tf.Variable(self.pretrained_model[name][0], name='weights', trainable=not lock)
                    if lock:
                        print("[!]\tLocked ", name, " parameters")
            else:
                kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([kernel_size, kernel_size, inputs.get_shape().as_list()[3], filters]), name='weights')

            #   save kernel and bias
            self.var_dict[(name, 0)] = kernel

            #   Collect Regularization Loss
            if lock is False and regularizers is not None:
                for regularizer in regularizers:
                    regularizer.collect(kernel)

            conv = tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1], padding=pad, data_format='NHWC')
            norm = self._batch_norm(conv, decay=0.9, epsilon=1e-3, use_loaded=use_loaded, lock=lock, name=name, is_training=self.training)
            relu = tf.nn.relu(norm)
            if self.w_summary:
                self.summ_histogram_list.append(tf.summary.histogram(name + 'weights', kernel, collections=['weight']))
            return relu
    
    def _convdw_bn_relu(self, inputs, kernel_size, strides, pad, name="_convdw_bn_relu", regularizers=[], use_loaded=False, lock=False):
        """ Depthwise Spatial Convolution (CONV2D) + BatchNormalization + ReLU Activation
        Args:
            inputs			: Input Tensor (Data Type : NHWC)
            kernel_size	    : Size of kernel
            strides		    : Stride
            pad				: Padding Type (VALID/SAME) # DO NOT USE 'SAME' NETWORK BUILT FOR VALID
            name			: Name of the block
            use_loaded      : Use related name to find weight and bias
            lock            : Lock the layer so the parameter won't be optimized
        Returns:
            norm			: Output Tensor
        """
        with tf.variable_scope(name):
            if use_loaded:
                if not self.training:
                    #   TODO:   Assertion
                    kernel = tf.constant(self.pretrained_model[name][0], name='weights')
                    print("[!]\tLayer restored! name of ", name)
                else:
                    kernel = tf.Variable(self.pretrained_model[name][0], name='weights', trainable=not lock)
                    if lock:
                        print("[!]\tLocked ", name, " parameters")
            else:
                kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([kernel_size, kernel_size, inputs.get_shape().as_list()[3], 1]), name='weights')

            #   save kernel and bias
            self.var_dict[(name, 0)] = kernel

            #   Collect Regularization Loss
            if lock is False and regularizers is not None:
                for regularizer in regularizers:
                    regularizer.collect(kernel)

            conv = tf.nn.depthwise_conv2d(inputs, kernel, [1, strides, strides, 1], padding=pad, data_format='NHWC')
            norm = self._batch_norm(conv, decay=0.9, epsilon=1e-3, use_loaded=use_loaded, lock=lock, name=name, is_training=self.training)
            relu = tf.nn.relu(norm)
            if self.w_summary:
                self.summ_histogram_list.append(tf.summary.histogram(name + 'weights', kernel, collections=['weight']))
            return relu

    def _separable_conv(self, inputs, filters, kernel_size, stride, pad, name="_separable_conv", regularizers=[], use_loaded=False, lock=False):
        """ Separable 2D Convolution
        Args:
            inputs			: Input Tensor (Data Type : NHWC)
            filters		    : Number of filters (channels)
            kernel_size     : Size of Kernel
            strides		    : Stride
            pad				: Padding Type (VALID/SAME) # DO NOT USE 'SAME' NETWORK BUILT FOR VALID
            name			: Name of the block
            use_loaded      : Use related name to find weight and bias
            lock            : Lock the layer so the parameter won't be optimized
        """
        #   depthwise kernel has the same channel size as the input's channel
        net = self._convdw_bn_relu(inputs, kernel_size, stride, pad, name + '_depthwise', regularizers=regularizers, use_loaded=use_loaded, lock=lock)
        #   point-wise should only use 1_1 kernel and 1 stride
        net = self._conv_bn_relu(net, filters, 1, 1, pad, name + '_pointwise', regularizers=regularizers, use_loaded=use_loaded, lock=lock)
        return net

    def _conv_block(self, inputs, numOut, name='conv_block'):
        """ Convolutional Block
        Args:
            inputs	: Input Tensor
            numOut	: Desired output number of channel
            name	: Name of the block
        Returns:
            conv_3	: Output Tensor
        """
        with tf.variable_scope(name):
            with tf.variable_scope('norm_1'):
                norm_1 = tf.contrib.layers.batch_norm(inputs, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu, is_training=self.training)
                conv_1 = self._conv(norm_1, int(numOut / 2), kernel_size=1, strides=1, pad='VALID', name='conv1')
            with tf.variable_scope('norm_2'):
                norm_2 = tf.contrib.layers.batch_norm(conv_1, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu, is_training=self.training)
                pad = tf.pad(norm_2, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]), name='pad')
                conv_2 = self._conv(pad, int(numOut / 2), kernel_size=3, strides=1, pad='VALID', name='conv2')
            with tf.variable_scope('norm_3'):
                norm_3 = tf.contrib.layers.batch_norm(conv_2, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu, is_training=self.training)
                conv_3 = self._conv(norm_3, int(numOut), kernel_size=1, strides=1, pad='VALID', name='conv3')
            return conv_3
                
    def _skip_layer(self, inputs, numOut, name='skip_layer'):
        """ Skip Layer
        Args:
            inputs	: Input Tensor
            numOut	: Desired output number of channel
            name	: Name of the bloc
        Returns:
            Tensor of shape (None, inputs.height, inputs.width, numOut)
        """
        with tf.variable_scope(name):
            if inputs.get_shape().as_list()[3] == numOut:
                return inputs
            else:
                conv = self._conv(inputs, numOut, kernel_size=1, strides=1, name='conv_sk')
                return conv				
    
    def _residual(self, inputs, numOut, name='residual_block'):
        """ Residual Unit
        Args:
            inputs	: Input Tensor
            numOut	: Number of Output Features (channels)
            name	: Name of the block
        """
        with tf.variable_scope(name):
            convb = self._conv_block(inputs, numOut, name='_conv_bl')
            skipl = self._skip_layer(inputs, numOut, name='_conv_sk')
            if self.net_debug:
                return tf.nn.relu(tf.add_n([convb, skipl], name='res_block'))
            else:
                return tf.add_n([convb, skipl], name='res_block')
                    
    def crop_and_resize(self, image, boxes, box_ind, crop_size, pad_border=True):
        """
        Aligned version of tf.image.crop_and_resize, following our definition of floating point boxes.

        Args:
            image: NHWC
            boxes: nx4, x1y1x2y2
            box_ind: (n,)
            crop_size (int):
        Returns:
            n,size,size,C
        """
        assert isinstance(crop_size, int), crop_size
        boxes = tf.stop_gradient(boxes)

        # TF's crop_and_resize produces zeros on border
        if pad_border:
            # this can be quite slow
            image = tf.pad(image, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='SYMMETRIC')
            boxes = boxes + 1

        def transform_fpcoor_for_tf(boxes, image_shape, crop_shape):
            x0, y0, x1, y1 = tf.split(boxes, 4, axis=1)

            spacing_w = (x1 - x0) / tf.to_float(crop_shape[1])
            spacing_h = (y1 - y0) / tf.to_float(crop_shape[0])

            nx0 = (x0 + spacing_w / 2 - 0.5) / tf.to_float(image_shape[1] - 1)
            ny0 = (y0 + spacing_h / 2 - 0.5) / tf.to_float(image_shape[0] - 1)

            nw = spacing_w * tf.to_float(crop_shape[1] - 1) / tf.to_float(image_shape[1] - 1)
            nh = spacing_h * tf.to_float(crop_shape[0] - 1) / tf.to_float(image_shape[0] - 1)

            return tf.concat([ny0, nx0, ny0 + nh, nx0 + nw], axis=1)

        image_shape = tf.shape(image)[1:3]
        boxes = transform_fpcoor_for_tf(boxes, image_shape, [crop_size, crop_size])
        ret = tf.image.crop_and_resize(
            image, boxes, tf.to_int32(box_ind),
            crop_size=[crop_size, crop_size])
        return ret

    def roi_align(self, featuremap, boxes, box_ind, resolution, out_channel, name="RoIAlign"):
        """
        Args:
            featuremap: batchsize x H x W x C
            boxes: batchsize * [Nx4 floatbox]
            resolution: output spatial resolution

        Returns:
            N x res x res x C
        """
        # sample 4 locations per roi bin
        with tf.variable_scope(name):
            ret = self.crop_and_resize(
                        featuremap, 
                        self.cwh2tlbr(boxes),
                        box_ind,
                        resolution * 2)
            ret = tf.nn.avg_pool(ret, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', data_format='NHWC')
            ret = tf.reshape(ret, [-1, resolution, resolution, out_channel])
            return ret

    def bbox_transform_inv(self, boxes, deltas, name="bbox_transform_inverse"):
        """ transform bbox with deltas(offsets)
        NOTE:   the bounding box format is in cx, cy, w, h
        """
        with tf.variable_scope(name):
            pred_ctr_x = tf.add(tf.multiply(deltas[:, :, :, :, 0], boxes[:, :, :, :, 2]), boxes[:, :, :, :, 0])
            pred_ctr_y = tf.add(tf.multiply(deltas[:, :, :, :, 1], boxes[:, :, :, :, 3]), boxes[:, :, :, :, 1])
            pred_w = tf.multiply(tf.exp(deltas[:, :, :, :, 2]), boxes[:, :, :, :, 2])
            pred_h = tf.multiply(tf.exp(deltas[:, :, :, :, 3]), boxes[:, :, :, :, 3])

            return tf.stack([pred_ctr_x, pred_ctr_y, pred_w, pred_h], axis=-1) 
    
    def proposal_layer(self, score_map, bbox_offset, pos_map_single, anchors, out_size, nms_iou_threshold=0.5, pre_nms_resrv=200, post_nms_resrv=16, nms_score_threshold=float('-inf'), name="ProposalLayer"):
        """ extract bbox from score map
        NOTE:   return format shape in [batch_num, num_true, (cx, cy, w, h) 
        Args:
            score_map       :   shape of (batch_size, width, height, 2*anchors)
            bbox_offset     :   offset map
            pos_map         :   shape of (batch_size, width, height, 2) (re-use)
            anchors         :   list of anchors
            out_size        :   output size (like int(46))
            nms_iou_threshold   :   Non Maximum Suppression score threshold
            pre_nms_resrv   :   Number of record that the layer reserved before NMS
            post_nms_resrv  :   Number of record that the layer reserved after NMS

        Return:
            layer_out       :   list of tensor in different shapes

        NOTE:
            To implement with dynamic batch_size we use a rebundant channel to store all node
            We declare a pre-define style network structure, and control with the tf.shape
            with input and output

            Be aware of tf.shape, it can generate tensor of unknown shapes, and generate dence
            tensor in graph. Just take care of it after pad using tf.shape
        """
        with tf.variable_scope(name):
            with tf.variable_scope("Tensor_Reshape"):
                #   (batch_size, out_size, out_size, anchors, 4)
                bbox_offset = tf.zeros(tf.shape(bbox_offset), tf.float32)
                ex_bbox_offset = tf.reshape(bbox_offset, (-1, out_size, out_size, len(anchors), 4))
                #   multiply pos_map to batch_size
                pos_map = tf.tile(pos_map_single, multiples=[tf.shape(score_map)[0], 1, 1, 1, 1])
                #   transform_inv(pos_map, offset) = abs_position
                ex_bbox_offset = self.bbox_transform_inv(pos_map, ex_bbox_offset)
                #   (batch_size, out_size * out_size * anchors， 4)
                ex_bbox_offset = tf.reshape(ex_bbox_offset, (tf.shape(score_map)[0], out_size * out_size * len(anchors), 4))


                #   positive only <class num 1>
                #   (batch_size, out_size * out_size, anchors)
                score_map = tf.reshape(score_map, [-1, out_size*out_size*len(anchors), 2])
                
                score = score_map[:, :, 1]
                #   (batch_size, out_size * out_size * anchors)
            with tf.variable_scope("Expand"):
                #   expand to defined batch_size
                ex_score_pad = tf.pad(score, [[0, self.batch_size-tf.shape(score_map)[0]],[0,0]])
                ex_score_pad = tf.reshape(ex_score_pad, (self.batch_size, out_size * out_size * len(anchors)))
                ex_bbox_offset_pad = tf.pad(ex_bbox_offset, [[0, self.batch_size-tf.shape(score_map)[0]],[0,0],[0,0]])

            #   Create nodes for every batch in mini-batch
            #   And you need to collect those Nodes
            #   layer_out have length of batch_size
            layer_out = None
            with tf.variable_scope("RoI_Extract"):
                for idx in range(self.batch_size):
                    with tf.variable_scope("TopK_NMS_Cell"):
                        batch_ind = idx * tf.ones((out_size*out_size*len(anchors)))

                        #   Find Top-k element index
                        _, topk_indx = tf.nn.top_k(ex_score_pad[idx], k=pre_nms_resrv, sorted=True)

                        #   Collect all value (bbox, score, batch_indx)
                        topk_bbox = tf.gather(ex_bbox_offset_pad[idx], topk_indx)
                        topk_score = tf.gather(ex_score_pad[idx], topk_indx)
                        topk_batch_ind = tf.gather(batch_ind, topk_indx)

                        #   Get the index after doing NMS
                        topk_idx_nms = tf.image.non_max_suppression(self.cwh2tlbr_rev(topk_bbox),
                                                                    topk_score,
                                                                    post_nms_resrv,
                                                                    iou_threshold=nms_iou_threshold,
                                                                    score_threshold=nms_score_threshold)

                        #   Collect result after NMS
                        #   (num_true, 4)
                        nms_bbox = tf.gather(topk_bbox, topk_idx_nms)
                        nms_score = tf.gather(topk_score, topk_idx_nms)
                        #   (num_true, 1)
                        nms_batch_ind = tf.expand_dims(tf.gather(topk_batch_ind, topk_idx_nms), axis=-1)
                        #   (num_true, 5) [cx, cy, w, h, batch_ind]
                        t = tf.concat([nms_bbox, nms_batch_ind], -1)

                        if layer_out is None:
                            layer_out = t
                        else:
                            layer_out = tf.concat([layer_out, t], 0)
                #   return at input batch_size
                #   (num_true_all * 5)
                return layer_out

    def cwh2tlbr(self, bbox, name="CWH2TLBR_GATE"):
        """ Cx, Cy, W, H to TopLeft BottomRight
        """
        with tf.variable_scope(name):
            cx, cy, w, h = tf.split(bbox, 4, axis=-1)
            x1 = cx - w/2
            y1 = cy - h/2
            x2 = cx + w/2
            y2 = cy + h/2
            return tf.concat([x1, y1, x2, y2], axis=-1, name=name+"_out")

    def cwh2tlbr_rev(self, bbox, name="CWH2TLBR_GATE"):
        """ Cx, Cy, W, H to TopLeft BottomRight
        """
        with tf.variable_scope(name):
            cx, cy, w, h = tf.split(bbox, 4, axis=-1)
            x1 = cx - w/2
            y1 = cy - h/2
            x2 = cx + w/2
            y2 = cy + h/2
            return tf.concat([y1, x1, y2, x2], axis=-1, name=name+"_out")

    def tlbr2cwh(self, bbox, name="TLBR2CWH_GATE"):
        """ Cx, Cy, W, H to TopLeft BottomRight
        """
        with tf.variable_scope(name):
            x1, y1, x2, y2 = tf.split(bbox, 4, axis=-1)
            w = tf.abs(x2 - x1)
            h = tf.abs(y2 - y1)
            cx = x1 + w/2
            cy = y1 + h/2
            return tf.concat([cx, cy, w, h], axis=-1, name=name+"_out")

    def patch_with_crop_and_resize(self, img, s_bbox, out_size, target_size, name="ReProjGate"):
        """ back projection according to bounding box
        Args:
            img     :   img has shape of (h, w, c)
            s_bbox  :   bounding box shape of (4,) [cx, cy, w, h]
        
        TODO:   This Layer should output lossless target size output but now we just uses
                upsampled output. This need to be fixed in the future!!
        """
        with tf.variable_scope(name):
            #   first resize the image to its origin size
            with tf.variable_scope(name+"_image_resize"):
                s_bbox = tf.cast(s_bbox, tf.int32)
                s_mask = tf.cond(
                                tf.logical_and(tf.greater(s_bbox[2],0), tf.greater(s_bbox[3], 0)),
                                lambda: tf.image.resize_images(img, tf.stack([s_bbox[3], s_bbox[2]])),
                                lambda: tf.zeros([out_size, out_size, tf.shape(img)[-1]])
                                )

            #   convert to top left - bottom right format
            s_bbox_tlbr = self.cwh2tlbr(s_bbox, name=name+"_CWH2TLBR")

            #   then crop the valid area on origin image
            with tf.variable_scope(name+"_image_crop"):
                crop_box = tf.concat([tf.nn.relu(-s_bbox_tlbr[:2]), tf.nn.relu(s_bbox[2:4]-tf.nn.relu(s_bbox_tlbr[2:4]-(out_size, out_size)))], 0)
                #crop_box = tf.cast(crop_box, tf.int32)
                crop_box = tf.concat([tf.minimum(crop_box[:2], crop_box[2:4]), tf.maximum(crop_box[:2], crop_box[2:4])], 0)
                s_mask = s_mask[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
                s_mask = tf.cond(tf.logical_and(tf.greater(tf.shape(s_mask)[0],0), tf.greater(tf.shape(s_mask)[1],0)),
                                    lambda: s_mask,
                                    lambda: tf.zeros([out_size, out_size, tf.shape(img)[-1]], dtype=tf.float32)
                                    )
            #   finally pad the output into original size
            #   s_mask has shape of (out_size, out_size)
            with tf.variable_scope(name+"_image_pad"):
                padd_left = tf.nn.relu(s_bbox_tlbr[0])
                padd_right = tf.nn.relu(out_size-s_bbox_tlbr[2])
                padd_top = tf.nn.relu(s_bbox_tlbr[1])
                padd_bottom = tf.nn.relu(out_size-s_bbox_tlbr[3])
                padd = tf.stack([[padd_top, padd_bottom], [padd_left, padd_right], [0, 0]], name="padd_param")
                s_mask = tf.pad(s_mask, padd)
            s_mask = tf.image.resize_images(s_mask,(target_size, target_size))
            '''
            '''
            return s_mask

    def dispatch_layer(self, bboxes, masks, batch_ind, batch_size, rois_max, out_size, name="DispatchLayer"):
        """ Dispatch Layer
            dispatch every instance into different channel
        Args:
            bboxes      :   bboxes from PPN (batch * rois, 4) in format of (cx, cy, w, h)
            masks       :   masks from Mask Net (batch * rois, size, size, channel)
            batch_ind   :   batch_ind from PPN (batch * rois)
            batch_size  :   images in a mini-batch

        """
        with tf.variable_scope(name):
            #   re-map the segmentation to origin image
            #   match bbox & segmentation, then crop
            #   1. first we need to pad the input to batch_size * rois
            with tf.variable_scope('ReProjection_Layer'):
                bboxes_pad = tf.pad(bboxes, [[0, tf.constant(batch_size*rois_max)-tf.shape(bboxes)[0]],[0,0]])
                bboxes_pad = tf.cast(bboxes_pad, tf.int32)
                bboxes_pad = tf.reshape(bboxes_pad, [batch_size*rois_max, 4])
                masks_pad = tf.pad(masks, [[0, tf.constant(batch_size*rois_max)-tf.shape(bboxes)[0]],[0,0],[0,0],[0,0]])
                mask_pad = tf.reshape(mask_pad, [batch_size*rois_max, out_size, out_size, channel])
                fan_in = None
                for n in range(batch_size * rois_max):
                    #   s_mask has shape of (height, width, channel)
                    s_mask = self.patch_with_crop_and_resize(
                                    masks_pad[n], 
                                    bboxes_pad[n],
                                    out_size,
                                    target_size,
                                    name="ReProjGate_"+str(n))
                    if fan_in is None:
                        fan_in = tf.expand_dims(s_mask, axis=0)
                    else:
                        fan_in = tf.concat([fan_in, tf.expand_dims(s_mask, axis=0)], 0)
                #   (rois, out_size, out_size, channel)
                #   fan_in has shape rois * height * width * channel
                fan_in = fan_in[:tf.shape(bboxes)[0]]

            #   2. secondly, we match masks to each batch
            with tf.variable_scope("Dispatch_Layer"):
                batch_mask = None
                for n in range(batch_size):
                    ind = tf.where(tf.equal(batch_ind, n))
                    ind = tf.squeeze(ind)
                    #   t has shape of (instance channel, height, width, channel)
                    #   if there is no RoI in this batch, fill with zeros
                    t = tf.cond(tf.greater(tf.squeeze(tf.shape(ind)), 0),
                                lambda: tf.gather(fan_in, ind),
                                lambda: tf.zeros(tf.shape(fan_in)))
                    #   if there are spare channels, fill them
                    t = tf.cond(tf.less(tf.shape(t)[0], rois_max), 
                                lambda: tf.pad(t, [[0,rois_max-tf.shape(t)[0]],[0,0],[0,0],[0,0]]),
                                lambda: t[:rois_max])
                    if batch_mask is None:
                        batch_mask = tf.expand_dims(t, axis=0)
                    else:
                        batch_mask = tf.concat([batch_mask, tf.expand_dims(t, axis=0)], 0)
            #   (batch_size, out_size, out_size, instance channel, channel)
            batch_mask = tf.transpose(batch_mask, perm=[0, 2, 3, 4, 1])
            batch_mask = tf.reshape(batch_mask, [channel, batch_size, out_size, out_size, rois_max])
            return batch_mask, fan_in

    def instance_layer(self, joint_map, ):
        pass


    '''
    def nms(self, dets, thresh, force_cpu=False, device_id=0):
        """ Custom Non Maxium Suppresion
        Dispatch to either CPU or GPU NMS implementations.
        """
        if dets.shape[0] == 0:
            return []
        if not force_cpu:
            return gpu_nms(dets, thresh, device_id=device_id)
        else:
            return cpu_nms(dets, thresh)
    '''