#coding: utf-8
import numpy as np
import cv2
import tensorflow as tf

class Regularizer(object):
    """
    Regularization collectors
    """
    def __init__(self, beta=0.00001, name="Regularizer"):
        self.name = name
        self.beta = beta
        self.norm_list = []

    def __regularizing_function__(self, vector):
        """
        Accept only 1-d vector
        """
        raise NotImplementedError
    
    def collect(self, weights):
        """
        Collect any input weights to its list
        """
        vectorized_weright = tf.reshape(weights, [-1])
        norm = self.__regularizing_function__(vectorized_weright)
        self.norm_list.append(norm)

    def __call__(self):
        return self.beta * tf.reduce_mean(self.norm_list)
    
class L2Regularizer(Regularizer):
    """
    L2 Regularization
    """
    def __regularizing_function__(self, vector):
        """
        sum(x ** 2) / 2
        """
        return tf.nn.l2_loss(vector)

class L1Regularizer(Regularizer):
    """
    L2 Regularization
    """
    def __regularizing_function__(self, vector):
        """
        sum(x)
        """
        return tf.reduce_sum(vector)

