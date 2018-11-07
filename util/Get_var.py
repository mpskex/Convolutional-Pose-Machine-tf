#coding: utf-8
import tensorflow as tf
import numpy as np
from tensorflow.python import pywrap_tensorflow   

""" This script is to extract parameters from a tensorflow checkpoints
    (MobileNet v1)

    Convolution weight              :           weights             0
    Batch Normalization Beta        :           beta(offset)        1
    Batch Normalization Gamma       :           gamma(scale)        2
    Batch Normalization Moving Mean :           movmean             3
    Batch Normalization Mov Var     :           movvar              4

    mpsk@github
"""

def save_npy(var_dict, save_path):
    """ Save the parameters

    :param save_path:       path to save
    :return:
    """
    data_dict = {}
    for (name, idx), var in var_dict.items():
        if name not in data_dict:
            data_dict[name] = {}
        print("[*]\tCreating dict for layer ", name, "-", str(idx))
        data_dict[name][idx] = var
    np.save(save_path, data_dict)
    print("[*]\tfile saved to", save_path)

def ExtractParams(meta_path, ckpt_path):
    """
    var_dict = {}
    with tf.device('/job:ps/task:4'):
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(meta_path)
            saver.restore(sess, ckpt_path)
            variable_name = [v.name for v in tf.trainable_variables()]
            for name in variable_name:
                print name
    """
    var_dict = {}
    pos_dict = {'weights':0,
                'depthwise_weights':0,
                'BatchNorm/beta':1,
                'BatchNorm/gamma':2,
                'BatchNorm/moving_variance':3,
                'BatchNorm/moving_mean':4}
    reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)  
    var_to_shape_map = reader.get_variable_to_shape_map()  
    for key in var_to_shape_map:
        if key.split('/')[0] == 'MobilenetV1':
            fullname = key.split('/')[1]
            #if fullname in ['Conv2d_9_depthwise']:#,'Conv2d_9_pointwise']:
            layername = ' '.join(fullname.split('_')[:2])
            jointname = fullname.split('_')[-1]
            if '/'.join(key.split('/')[2:]) in pos_dict.keys():
                var_dict[(fullname, pos_dict['/'.join(key.split('/')[2:])])] = reader.get_tensor(key)
                print(reader.get_tensor(key).shape)
            #print("tensor_name: %s %s %s" %(layername, jointname, key))  
    return var_dict
if __name__ == '__main__':
    var_dict = ExtractParams("../model/mobilenet_v1_1.0_224.ckpt.meta", "model/mobilenet_v1_1.0_224.ckpt")
    save_npy(var_dict, '../model/mobilev1_1.0_converted.npy')
