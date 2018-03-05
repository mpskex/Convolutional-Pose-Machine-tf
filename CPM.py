#coding: utf-8
import os
import urllib
import numpy as np
import scipy.io as sio
import tensorflow.contrib.layers as layers
import tensorflow as tf

import model
#from dataset import Dataset
import datagen

class CPM():
    """
    CPM net
    """
    def __init__(self, base_lr=0.0005, in_size=368, batch_size=16, epoch=200, dataset = None, log_dir=None, stage=6, epoch_size=1000):
        tf.reset_default_graph()
        self.sess = tf.Session()
        if log_dir:
            self.writer = tf.summary.FileWriter(log_dir)
        self.log_dir = log_dir

        self.dataset = dataset
        self.joint_num = 16
        self.stage = stage

        self.base_lr = base_lr
        self.in_size = in_size
        self.batch_size = batch_size
        self.epoch = epoch
        self.epoch_size = epoch_size
        self.dataset = dataset
        #   step learning rate policy
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(base_lr,
            self.global_step, 2000, 0.333,
            staircase=True)
        #self.learning_rate = base_lr
        self.train_step = []
        self.losses = []

        self.img = None
        self.gtmap = None
        self.stagehmap = []

        self.summ_scalar_list = []
        self.summ_image_list = []


    def __build_ph(self):
        #   Valid & Train input
        #   input image : channel 3
        self.img = tf.placeholder(tf.float32, 
            shape=[None, self.in_size, self.in_size, 3], name="img_in")
        #   input center map : channel 1 (downscale by 8)
        self.weight = tf.placeholder(tf.float32,
            shape=[None, self.in_size/8, self.in_size, self.in_size])
        
        #   Train input
        #   input ground truth : channel 1 (downscale by 8)
        self.gtmap = tf.placeholder(tf.float32, 
            shape=[None, self.in_size/8, self.in_size/8, self.joint_num+1], name="gtmap")
        print "- PLACEHOLDER build finished!"
    
    def __build_train_op(self):
        #   Optimizer
        with tf.name_scope('loss'):
            for idx in range(len(self.stagehmap)):
                __para = []
                assert self.stagehmap!=[]
                loss = tf.multiply(self.weight, 
                    tf.reduce_sum(tf.nn.l2_loss(
                        self.stagehmap[idx] - self.gtmap, name='loss_stage_%d' % idx))))
                self.losses.append(loss)
                self.summ_scalar_list.append(tf.summary.scalar("loss in stage"+str(idx+1),
                     loss))
            self.total_loss = tf.reduce_mean(self.losses)
            self.summ_scalar_list.append(tf.summary.scalar("total loss", self.total_loss))
            self.summ_scalar_list.append(tf.summary.scalar("lr", self.learning_rate))
            print "- LOSS & SCALAR_SUMMARY build finished!"
        with tf.name_scope('optimizer'):
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                #self.optimizer = tf.train.AdamOptimizer(self.learning_rate, epsilon=1e-8)
                #self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
                self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
                #   Global train
                self.train_step.append(self.optimizer.minimize(self.total_loss/self.batch_size, 
                    global_step=self.global_step,
                    colocate_gradients_with_ops=True))
        print "- OPTIMIZER build finished!"

    def BuildModel(self):
        #   input
        with tf.name_scope('input'):
            self.__build_ph()
        #   assertion
        assert self.img!=None and self.gtmap!=None
        #   the net
        self.stagehmap = self.net(self.img)
        
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
        #   merge all summary
        self.summ_image = tf.summary.merge(self.summ_image_list)
        self.summ_scalar = tf.summary.merge(self.summ_scalar_list)
        self.writer.add_graph(self.sess.graph)

    def train(self):
        _epoch_count = 0
        _iter_count = 0
    
        #   datagen from Hourglass
        self.generator = self.dataset._aux_generator(self.batch_size, normalize = True, sample_set = 'train')
        self.valid_gen = self.dataset._aux_generator(self.batch_size, normalize = True, sample_set = 'valid')

        for n in range(self.epoch):
            for m in range(self.epoch_size):
                #   datagen from hourglass
                _train_batch = next(self.generator)
                '''
                #   origin dataset
                _train_batch = self.dataset.GenerateOneBatch()
                '''
                #'''
				#   datagen from hourglass
                _train_batch = next(self.generator)[:3]
                #'''
                print "[*] small batch generated!"
                for step in self.train_step:
                    self.sess.run(step, feed_dict={self.img: _train_batch[0],
                        self.gtmap:_train_batch[1]})
                #   summaries
                if _iter_count % 20 == 0:
                    #'''
                    self.writer.add_summary(
                        self.sess.run(self.summ_image,feed_dict={self.img: _train_batch[0], self.gtmap:_train_batch[1], self.weight:_train_batch[2]}))
                    #'''
                    maps = self.sess.run(self.stagehmap,
                        feed_dict={self.img: _train_batch[0],
                                    self.gtmap:_train_batch[1],
                                    self.weight:_train_batch[2]})
                    for i in range(len(maps)):
                        print "[!] saved heatmap with size of ", maps[i].shape
                        np.save(self.log_dir+"stage"+str(i+1)+"map.npy",
                            maps[i])
                    gt = self.sess.run(self.gtmap,
                        feed_dict={self.img: _train_batch[0],
                                    self.gtmap:_train_batch[1],
                                    self.weight:_train_batch[2]})
                    print "[!] saved ground truth with size of ", gt.shape
                    np.save(self.log_dir+"gt.npy", gt)
                    del gt, maps
                    #'''
                if _iter_count % 10 == 0:
                    print "epoch ", _epoch_count, " iter ", _iter_count, self.sess.run(self.total_loss, feed_dict={self.img: _train_batch[0], self.gtmap:_train_batch[1], self.weight:_train_batch[2]})
                    self.writer.add_summary(
                        self.sess.run(self.summ_scalar,feed_dict={self.img: _train_batch[0], self.gtmap:_train_batch[1], self.weight:_train_batch[2]}),
                        _iter_count)
                print "iter:", _iter_count
                _iter_count += 1
                self.writer.flush()
            _epoch_count += 1
            #   save model every epoch
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
        for j in range(len(self.stagehmap)):
            for i in range(self.joint_num):
                self.summ_scalar_list.append(tf.summary.scalar("stage "+str(j)+"-"+str(i)+"th joint accuracy", self._accur(self.stagehmap[j][i], self.gtmap[i], self.batch_size), 'accuracy'))
        print "- ACC_SUMMARY build finished!"

    def __build_monitor(self):
        #   calculate the return full map
        __all_gt = tf.expand_dims(tf.expand_dims(tf.reduce_sum(tf.transpose(self.gtmap, perm=[0,3,1,2])[0], axis=[0]), 0), 3)
        __image = tf.expand_dims(tf.transpose(self.img, perm=[0,3,1,2])[0], 3)
        self.summ_image_list.append(tf.summary.image("gtmap", __all_gt,
            max_outputs=1))
        self.summ_image_list.append(tf.summary.image("image", __image,
            max_outputs=3))
        
        for m in range(len(self.stagehmap)):
            #   __sample_pred have the shape of
            #   16 * INPUT+_SIZE/8 * INPUT_SIZE/8
            __sample_pred = tf.transpose(self.stagehmap[m], perm=[0,3,1,2])[0]
            #   __all_pred have shape of 
            #   INPUT_SIZE/8 * INPUT_SIZE/8
            __all_pred = tf.expand_dims(tf.expand_dims(tf.reduce_sum(tf.transpose(self.stagehmap[m], perm=[0,3,1,2])[0], axis=[0]), 0), 3)
            print "\tvisual heat map have shape of ", __all_pred.shape
            self.summ_image_list.append(tf.summary.image("stage"+str(m)+" map", __all_pred, max_outputs=1))
        del __all_gt, __image, __sample_pred, __all_pred
        print "- IMAGE_SUMMARY build finished!"

    def net(self, image):
        return model.Net(image, self.joint_num, stage=self.stage)

    def __TestAcc(self):
        self.dataset.shuffle()
        assert self.dataset.idx_batches!=None
        for m in self.dataset.idx_batches:
            _train_batch = self.dataset.GenerateOneBatch()
            print "[*] small batch generated!"
            for i in range(self.joint_num):
                self.sess.run(tf.summary.scalar(i,self._accur(self.gtmap[i], self.gtmap[i], self.batch_size), 'accuracy'))


