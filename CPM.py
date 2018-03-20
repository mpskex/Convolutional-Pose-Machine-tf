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

class CPM():
    """
    CPM net
    """
    def __init__(self, base_lr=0.0005, in_size=368, batch_size=16, epoch=20, dataset = None, log_dir=None, stage=6,
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
        self.batch_size = batch_size
        self.epoch = epoch
        self.epoch_size = epoch_size
        self.dataset = dataset

        #   step learning rate policy
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(base_lr,
            self.global_step, 10000, 0.333,
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
        else:
            self.pretrained_model = None
        #   dictionary of network parameters
        self.var_dict = {}

        #   predict mode
        self.predict = predict


    def __build_ph(self):
        """ Building Placeholder in tensorflow session
        :return:
        """
        #   Valid & Train input
        #   input image : channel 3
        self.img = tf.placeholder(tf.float32, 
            shape=[None, self.in_size, self.in_size, 3], name="img_in")
        #   input center map : channel 1 (downscale by 8)
        self.weight = tf.placeholder(tf.float32 ,
            shape=[None, self.joint_num+1])

        #   Train input
        #   input ground truth : channel 1 (downscale by 8)
        self.gtmap = tf.placeholder(tf.float32, 
            shape=[None, self.stage, self.in_size/8, self.in_size/8, self.joint_num+1], name="gtmap")
        print "- PLACEHOLDER build finished!"
    
    def __build_train_op(self):
        """ Building training associates: losses & loss summary
        :return:
        """
        #   Optimizer
        with tf.name_scope('loss'):
            __para = []
            #'''
            loss = tf.multiply(self.weight,
                tf.reduce_sum(tf.nn.l2_loss(
                    self.output - self.gtmap, name='loss_final')))
            self.losses.append(loss)
            #'''
            '''
            if self.w_loss:
                self.loss = tf.reduce_mean(self.weighted_bce_loss(), name='reduced_loss')
            else:
                self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, 
                    labels= self.gtMaps), name= 'cross_entropy_loss')
            '''
            self.total_loss = tf.reduce_mean(self.losses)
            self.summ_scalar_list.append(tf.summary.scalar("total loss", self.total_loss))
            self.summ_scalar_list.append(tf.summary.scalar("lr", self.learning_rate))
            print "- LOSS & SCALAR_SUMMARY build finished!"
        with tf.name_scope('optimizer'):
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                #self.optimizer = tf.train.AdamOptimizer(self.learning_rate, epsilon=1e-8)
                #self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
                self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
                #self.optimizer = tf.train.AdamOptimizer(self.learning_rate, epsilon=1e-8)
                #   Global train
                self.train_step.append(self.optimizer.minimize(self.total_loss/self.batch_size,
                    global_step=self.global_step))
        print "- OPTIMIZER build finished!"

    def BuildModel(self):
        """ Building model in tensorflow session
        :return:
        """
        #   input
        with tf.name_scope('input'):
            self.__build_ph()
        #   assertion
        assert self.img!=None and self.gtmap!=None
        #   the net
        self.output = self.net(self.img)
        if not self.predict:
            #   train op
            with tf.name_scope('train'):
                self.__build_train_op()
            with tf.name_scope('image_summary'):
                self.__build_monitor()
            with tf.name_scope('accuracy'):
                self.__build_accuracy()
        #   initialize all variables
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        if not self.predict:
            #   merge all summary
            self.summ_image = tf.summary.merge(self.summ_image_list)
            self.summ_scalar = tf.summary.merge(self.summ_scalar_list)
            self.summ_accuracy = tf.summary.merge(self.summ_accuracy_list)
            self.summ_histogram = tf.summary.merge(self.summ_histogram_list)
            self.writer.add_graph(self.sess.graph)
        print "[*]\tModel Built"

    def save_npy(self, save_path=None):
        """ Save the parameters
        :param save_path:       path to save
        :return:                int - 0 for success
        """
        if save_path == None:
            save_path = self.log_dir + 'model.npy'
        data_dict = {}
        for (name, idx), var in self.var_dict:
            var_out = self.sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out
        np.save(save_path, data_dict)
        print("[*]\tfile saved to", save_path)

    def restore_sess(self, model=None):
        """ Restore session from ckpt format file
        :param model:   model path like mode
        :return:        Nothing
        """
        if model is not None:
            t = time.time()
            self.saver.restore(self.sess, model)
            print("[*]\tSESS Restored!")
        else:
            print("Please input proper model path to restore!")
            raise ValueError


    def train(self):
        """ Training Progress in CPM
        :return:    Nothing to output
        """
        _epoch_count = 0
        _iter_count = 0
    
        #   datagen from Hourglass
        self.generator = self.dataset._aux_generator(self.batch_size, stacks=self.stage, normalize = True, sample_set = 'train')
        self.valid_gen = self.dataset._aux_generator(self.batch_size, stacks=self.stage, normalize = True, sample_set = 'valid')

        for n in range(self.epoch):
            for m in range(self.epoch_size):
                #   datagen from hourglass
                _train_batch = next(self.generator)
                print "[*] small batch generated!"
                for step in self.train_step:
                    self.sess.run(step, feed_dict={self.img: _train_batch[0],
                        self.gtmap:_train_batch[1],
                        self.weight:_train_batch[2]})
                #   doing save numpy params
                self.save_npy()
                #   summaries
                if _iter_count % 10 == 0:
                    _test_batch = next(self.valid_gen)
                    print "epoch ", _epoch_count, " iter ", _iter_count, self.sess.run(self.total_loss, feed_dict={self.img: _test_batch[0], self.gtmap:_test_batch[1], self.weight:_test_batch[2]})
                    #   doing the scalar summary
                    self.writer.add_summary(
                        self.sess.run(self.summ_scalar,feed_dict={self.img: _train_batch[0],
                                                        self.gtmap:_train_batch[1],
                                                        self.weight:_train_batch[2]}),
                        _iter_count)
                    self.writer.add_summary(
                        self.sess.run(self.summ_image, feed_dict={self.img: _test_batch[0],
                                                        self.gtmap:_test_batch[1],
                                                        self.weight:_test_batch[2]}),
                        _iter_count)
                    self.writer.add_summary(
                        self.sess.run(self.summ_accuracy, feed_dict={self.img: _test_batch[0],
                                                              self.gtmap: _test_batch[1],
                                                              self.weight: _test_batch[2]}),
                        _iter_count)
                    self.writer.add_summary(
                        self.sess.run(self.summ_histogram, feed_dict={self.img: _train_batch[0],
                                                        self.gtmap:_train_batch[1],
                                                        self.weight:_train_batch[2]}),
                        _iter_count)

                if _iter_count % 20 == 0:
                    #   generate heatmap from the network
                    maps = self.sess.run(self.output,
                            feed_dict={self.img: _test_batch[0],
                                    self.gtmap: _test_batch[1],
                                    self.weight: _test_batch[2]})
                    if self.log_dir is not None:
                        print "[!] saved heatmap with size of ", maps.shape
                        np.save(self.log_dir+"output.npy", maps)
                        print "[!] saved ground truth with size of ", self.gtmap.shape
                        np.save(self.log_dir+"gt.npy", _test_batch[1])
                    del maps, _test_batch
                print "iter:", _iter_count
                _iter_count += 1
                self.writer.flush()
                del _train_batch
            _epoch_count += 1
            #   save model every epoch
            if self.log_dir is not None:
                self.saver.save(self.sess, os.path.join(self.log_dir, "model.ckpt"), n)

    def _argmax(self, tensor):
        """ ArgMax
        Args:
            tensor	: 2D - Tensor (Height x Width : 64x64 )
        Returns:
            arg		: Tuple of maxlen(self.losses) position
        """
        resh = tf.reshape(tensor, [-1])
        argmax = tf.argmax(resh, 0)
        return (argmax // tensor.get_shape().as_list()[0], argmax % tensor.get_shape().as_list()[0])

    def _compute_err(self, u, v):
        """ Given 2 tensors compute the euclidean distance (L2) between maxima locations
        Args:
            u		: 2D - Tensor (Height x Width : 64x64 )
            v		: 2D - Tensor (Height x Width : 64x64 )
        Returns:
            (float) : Distance (in [0,1])
        """
        u_x,u_y = self._argmax(u)
        v_x,v_y = self._argmax(v)
        return tf.divide(tf.sqrt(tf.square(tf.to_float(u_x - v_x)) + tf.square(tf.to_float(u_y - v_y))), tf.to_float(91))

    def _accur(self, pred, gtMap, num_image):
        """ Given a Prediction batch (pred) and a Ground Truth batch (gtMaps),
        returns one minus the mean distance.
        Args:
            pred		: Prediction Batch (shape = num_image x 64 x 64)
            gtMaps		: Ground Truth Batch (shape = num_image x 64 x 64)
            num_image 	: (int) Number of images in batch
        Returns:
            (float)
        """
        err = tf.to_float(0)
        for i in range(num_image):
            err = tf.add(err, self._compute_err(pred[i], gtMap[i]))
        return tf.subtract(tf.to_float(1), err/num_image)

    def __build_accuracy(self):
        """ 
        Computes accuracy tensor
        """
        for i in range(self.joint_num):
            self.summ_accuracy_list.append(tf.summary.scalar(self.joints[i]+"_accuracy",
                                                           self._accur(self.output[:, self.stage-1, :, :, i], self.gtmap[:, self.stage-1, :, :, i], self.batch_size),
                                                           'accuracy'))
        print "- ACC_SUMMARY build finished!"

    def __build_monitor(self):
        """ Building image summaries
        :return:
        """
        with tf.device(self.cpu):
            #   calculate the return full map
            __all_gt = tf.expand_dims(tf.expand_dims(tf.reduce_sum(tf.transpose(self.gtmap, perm=[0, 1, 4, 2, 3])[0], axis=[0, 1]), 0), 3)
            self.summ_image_list.append(tf.summary.image("gtmap", __all_gt, max_outputs=1))
            self.summ_image_list.append(tf.summary.image("image", tf.expand_dims(self.img[0], 0), max_outputs=3))
            print "\t* monitor image have shape of ", tf.expand_dims(self.img[0], 0).shape
            print "\t* monitor GT have shape of ", __all_gt.shape
            for m in range(self.stage):
                #   __sample_pred have the shape of
                #   16 * INPUT+_SIZE/8 * INPUT_SIZE/8
                __sample_pred = tf.transpose(self.output[0, m], perm=[2, 0, 1])
                #   __all_pred have shape of
                #   INPUT_SIZE/8 * INPUT_SIZE/8
                __all_pred = tf.expand_dims(tf.expand_dims(tf.reduce_sum(__sample_pred, axis=[0]), 0), 3)
                print "\tvisual heat map have shape of ", __all_pred.shape
                self.summ_image_list.append(tf.summary.image("stage"+str(m)+" map", __all_pred, max_outputs=1))
            del __all_gt, __sample_pred, __all_pred
            print "- IMAGE_SUMMARY build finished!"

    def __TestAcc(self):
        """ Calculate Accuracy (Please use validation data)
        :return:
        """
        self.dataset.shuffle()
        assert self.dataset.idx_batches!=None
        for m in self.dataset.idx_batches:
            _train_batch = self.dataset.GenerateOneBatch()
            print "[*] small batch generated!"
            for i in range(self.joint_num):
                self.sess.run(tf.summary.scalar(i,self._accur(self.gtmap[i], self.gtmap[i], self.batch_size), 'accuracy'))

    def weighted_bce_loss(self):
        """ Create Weighted Loss Function
        WORK IN PROGRESS
        """
        self.bceloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, labels= self.gtmap), name= 'cross_entropy_loss')
        e1 = tf.expand_dims(self.weight,axis = 1, name = 'expdim01')
        e2 = tf.expand_dims(e1,axis = 1, name = 'expdim02')
        e3 = tf.expand_dims(e2,axis = 1, name = 'expdim03')
        return tf.multiply(e3,self.bceloss, name = 'lossW')

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

    #   ======= Net Component ========

    def _conv(self, inputs, filters, kernel_size = 1, strides = 1, pad='VALID', name='conv', use_loaded=False):
        """ Spatial Convolution (CONV2D)
        Args:
            inputs			: Input Tensor (Data Type : NHWC)
            filters		: Number of filters (channels)
            kernel_size	: Size of kernel
            strides		: Stride
            pad				: Padding Type (VALID/SAME) # DO NOT USE 'SAME' NETWORK BUILT FOR VALID
            name			: Name of the block
        Returns:
            conv			: Output Tensor (Convolved Input)
        """
        with tf.variable_scope(name):
            if use_loaded:
                if self.pretrained_model is not None:
                    if self.predict:
                        kernel = tf.constant(self.pretrained_model[name][0], name='weights')
                        bias = tf.constant(self.pretrained_model[name][1], name='bias')
                    else:
                        kernel = tf.Variable(self.pretrained_model[name][0], name='weights')
                        bias = tf.Variable(self.pretrained_model[name][1], name='bias')
                else:
                    raise ValueError
            else:
                # Kernel for convolution, Xavier Initialisation
                kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([kernel_size,kernel_size, inputs.get_shape().as_list()[3], filters]), name= 'weights')
                bias = tf.Variable(tf.zeros([filters]), name='bias')
            #   save kernel and bias
            self.var_dict[(name,0)] = kernel
            self.var_dict[(name,1)] = bias
            conv = tf.nn.conv2d(inputs, kernel, [1,strides,strides,1], padding=pad, data_format='NHWC')
            conv_bias = tf.nn.bias_add(conv, bias)
            if self.w_summary:
                with tf.device(self.cpu):
                    self.summ_histogram_list.append(tf.summary.histogram(name+'weights', kernel, collections=['weight']))
                    self.summ_histogram_list.append(tf.summary.histogram(name+'bias', bias, collections=['bias']))
            return conv_bias

    def _conv_bn_relu(self, inputs, filters, kernel_size = 1, strides=1, pad='VALID', name='conv_bn_relu', use_loaded=False, lock=False):
        """ Spatial Convolution (CONV2D) + BatchNormalization + ReLU Activation
        Args:
            inputs			: Input Tensor (Data Type : NHWC)
            filters		    : Number of filters (channels)
            kernel_size	    : Size of kernel
            strides		    : Stride
            pad				: Padding Type (VALID/SAME) # DO NOT USE 'SAME' NETWORK BUILT FOR VALID
            name			: Name of the block
            use_loaded      : Use related name to find weight and bias
        Returns:
            norm			: Output Tensor
        """
        with tf.variable_scope(name):
            if use_loaded:
                if  self.pretrained_model is not None:
                    if self.predict:
                        #   TODO:   Assertion
                        kernel = tf.constant(self.pretrained_model[name][0], name='weights')
                        bias = tf.constant(self.pretrained_model[name][1], name='bias')
                    else:
                        kernel = tf.Variable(self.pretrained_model[name][0], name='weights', trainable=not lock)
                        bias = tf.Variable(self.pretrained_model[name][1], name='bias', trainable=not lock)
                else:
                    raise ValueError
            else:
                kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([kernel_size,kernel_size, inputs.get_shape().as_list()[3], filters]), name= 'weights')
                bias = tf.Variable(tf.zeros([filters]), name='bias')
            #   save kernel and bias
            self.var_dict[(name,0)] = kernel
            self.var_dict[(name,1)] = bias
            conv = tf.nn.conv2d(inputs, kernel, [1,strides,strides,1], padding=pad, data_format='NHWC')
            conv_bias = tf.nn.bias_add(conv, bias)
            norm = tf.contrib.layers.batch_norm(conv_bias, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu, is_training = self.training)
            if self.w_summary:
                with tf.device(self.cpu):
                    self.summ_histogram_list.append(tf.summary.histogram(name+'weights', kernel, collections=['weight']))
                    self.summ_histogram_list.append(tf.summary.histogram(name+'bias', bias, collections=['bias']))
            return norm
    
    def _conv_block(self, inputs, numOut, name = 'conv_block'):
        """ Convolutional Block
        Args:
            inputs	: Input Tensor
            numOut	: Desired output number of channel
            name	: Name of the block
        Returns:
            conv_3	: Output Tensor
        """
        if self.tiny:
            with tf.name_scope(name):
                norm = tf.contrib.layers.batch_norm(inputs, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu, is_training = self.training)
                pad = tf.pad(norm, np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
                conv = self._conv(pad, int(numOut), kernel_size=3, strides=1, pad = 'VALID', name= 'conv')
                return conv
        else:
            with tf.variable_scope(name):
                with tf.name_scope('norm_1'):
                    norm_1 = tf.contrib.layers.batch_norm(inputs, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu, is_training = self.training)
                    conv_1 = self._conv(norm_1, int(numOut/2), kernel_size=1, strides=1, pad = 'VALID', name= 'conv')
                with tf.name_scope('norm_2'):
                    norm_2 = tf.contrib.layers.batch_norm(conv_1, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu, is_training = self.training)
                    pad = tf.pad(norm_2, np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
                    conv_2 = self._conv(pad, int(numOut/2), kernel_size=3, strides=1, pad = 'VALID', name= 'conv')
                with tf.name_scope('norm_3'):
                    norm_3 = tf.contrib.layers.batch_norm(conv_2, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu, is_training = self.training)
                    conv_3 = self._conv(norm_3, int(numOut), kernel_size=1, strides=1, pad = 'VALID', name= 'conv')
                return conv_3
                
    def _skip_layer(self, inputs, numOut, name = 'skip_layer'):
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
                conv = self._conv(inputs, numOut, kernel_size=1, strides = 1, name = 'conv')
                return conv				
    
    def _residual(self, inputs, numOut, name='residual_block'):
        """ Residual Unit
        Args:
            inputs	: Input Tensor
            numOut	: Number of Output Features (channels)
            name	: Name of the block
        """
        with tf.variable_scope(name):
            convb = self._conv_block(inputs, numOut)
            skipl = self._skip_layer(inputs, numOut)
            if self.net_debug:
                return tf.nn.relu(tf.add_n([convb, skipl], name = 'res_block'))
            else:
                return tf.add_n([convb, skipl], name = 'res_block')

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
                net = self._conv_bn_relu(feat_map, 256, 3, 1, 'SAME', 'conv4_3_CPM', use_loaded=self.load_pretrained)
                net = self._conv_bn_relu(net, 256, 3, 1, 'SAME', 'conv4_4_CPM', use_loaded=self.load_pretrained)
                net = self._conv_bn_relu(net, 256, 3, 1, 'SAME', 'conv4_5_CPM', use_loaded=self.load_pretrained)
                net = self._conv_bn_relu(net, 256, 3, 1, 'SAME', 'conv4_6_CPM', use_loaded=self.load_pretrained)
                net = self._conv_bn_relu(net, 128, 3, 1, 'SAME', 'conv4_7_CPM', use_loaded=self.load_pretrained)
                net = self._conv_bn_relu(net, 512, 1, 1, 'SAME', 'conv5_1_CPM', use_loaded=self.load_pretrained)
                net = self._conv(net, self.joint_num+1, 1, 1, 'SAME', 'conv5_2_CPM', use_loaded=self.load_pretrained)
                return net
            elif stage_num > 1:
                net = tf.concat([feat_map, last_stage], 3)
                net = self._conv_bn_relu(net, 128, 7, 1, 'SAME', 'Mconv1_stage'+str(stage_num), use_loaded=self.load_pretrained)
                net = self._conv_bn_relu(net, 128, 7, 1, 'SAME', 'Mconv2_stage'+str(stage_num), use_loaded=self.load_pretrained)
                net = self._conv_bn_relu(net, 128, 7, 1, 'SAME', 'Mconv3_stage'+str(stage_num), use_loaded=self.load_pretrained)
                net = self._conv_bn_relu(net, 128, 7, 1, 'SAME', 'Mconv4_stage'+str(stage_num), use_loaded=self.load_pretrained)
                net = self._conv_bn_relu(net, 128, 7, 1, 'SAME', 'Mconv5_stage'+str(stage_num), use_loaded=self.load_pretrained)
                net = self._conv_bn_relu(net, 128, 1, 1, 'SAME', 'Mconv6_stage'+str(stage_num), use_loaded=self.load_pretrained)
                net = self._conv(net, self.joint_num+1, 1, 1, 'SAME', 'Mconv7_stage'+str(stage_num), use_loaded=self.load_pretrained)
                return net

