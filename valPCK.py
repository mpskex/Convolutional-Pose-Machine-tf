#coding: utf-8
import cv2
import time
import numpy as np

import CPM
import predict
import Global
import datagen_mpii as datagen
import matplotlib.pyplot as plt

"""
    PCKh Validation
        For Single Person Pose Estimation
    Human Pose Estimation Project in Lab of IP
    Author: Liu Fangrui aka mpsk
        Beijing University of Technology
            College of Computer Science & Technology
    Experimental Code
        !!DO NOT USE IT AS DEPLOYMENT!!

    ATTENTION:
        A pre-gen Validation set is needed to do validation seperately
"""

def visualize_accuracy(a, start=0.01, end=0.5, showlist=None, resolution=100):
    """ Compute the Accuracy in rate
    Args:
        a:              distance matrix like err in compute_distance 
        showlist:       list of joints to be shown
        resolutional:   determine the 
    """
    if showlist is None:
        showlist = range(0, len(Global.joint_list))
    else:
        show_list = showlist

    plt.title("CPM PCKh benchmark on MPII")
    plt.xlabel("Normalized distance")
    plt.ylabel("Accuracy")

    dists = np.linspace(start, end, resolution)
    re = np.zeros((dists.shape[0], a.shape[1]), np.float)
    av = np.zeros((dists.shape[0]), np.float)
    for ridx in range(dists.shape[0]):
        print "[*]\tProcessing Result in normalized distance of", dists[ridx]
        for j in range(a.shape[1]):
            condition = a[:,j] <= dists[ridx]
            re[ridx, j] = len(np.extract(condition, a[:,j]))/float(a.shape[0])
            print "[*]\t", Global.joint_list[j], " Accuracy :\t\t", re[ridx, j]
        av[ridx] = np.average(re[ridx])
        print "[*]\t\tTOTAL Accuracy :\t\t", av[ridx]

    for j in show_list:
        plt.plot(dists, re[:, j], label=Global.joint_list[j], linewidth = 2.0)
    plt.plot(dists, av, label="Average",linewidth = 3.0)
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()


def compute_distance(model, dataset, metric='PCKh', debug=False):
    """ 
    Args:
        model:      model to load
        dataset:    dataset to generate image & ground truth
    Return:
        An normalized distance matrix with shape of num_of_img x joint_num
    """
    if metric == 'PCKh':
        normJ_a = 9
        normJ_b = 8
    elif metric == 'PCK':
        normJ_a = 12
        normJ_b = 3
    else:
        raise ValueError

    err = np.zeros((len(dataset.valid_set), len(Global.joint_list)), np.float)
    print "[*]\tCreated error matrix with shape of ", err.shape
    paral = 16
    if debug:
        paral = 2
    for _iter in range(len(dataset.valid_set)/paral):
        im_list = []
        j_list = []
        w_list = []

        for n in range(paral):
            img, new_j, w, joint_full, max_l = dataset.getSample(sample=dataset.valid_set[_iter*paral+n])
            im_list.append(img)
            j_list.append(new_j)
            w_list.append(w)
        
        w = np.array(w_list)
        j_gt = np.array(j_list)

        #   estimate by network
        j_dt,_ = predict.predict(im_list, model=model, thresh=0.0, debug=True)
        w = np.transpose(np.hstack((np.expand_dims(w,1), np.expand_dims(w,1))), axes=[0,2,1])
        assert j_dt.shape == j_gt.shape

        for n in range(j_dt.shape[0]):
            err[_iter*paral+n] = np.linalg.norm(w[n]*(j_gt[n,:,:]-j_dt[n,:,-1::-1]),axis=1) / np.linalg.norm(j_gt[n,normJ_a,:]-j_gt[n,normJ_b,:], axis=0)
        print "[*]\tTemp Error is ", np.average(err[_iter*paral:_iter*paral+paral], axis=0)
        if debug:
            return err

    aver_err = np.average(err)
    print "[*]\tAverage PCKh Normalised distance is ", aver_err

    return err

if __name__ == '__main__':
    print('--Creating Dataset')
    dataset = datagen.DataGenerator(Global.joint_list, Global.IMG_ROOT, Global.training_txt_file, remove_joints=None, in_size=Global.INPUT_SIZE)
    dataset._create_train_table()
    dataset.valid_set = np.load('Dataset-Validation-Set.npy')
    
    model = CPM.CPM(pretrained_model=None,
            #   enable if you want to do it in cpu only mode   
            #cpu_only=False,
            training=False)
    model.BuildModel()
    model.restore_sess('model/model.ckpt-99')

    dist = compute_distance(model, dataset, metric='PCKh')
    visualize_accuracy(dist, start=0.01, end=0.5, showlist=[0,1,4,5,10,11,14,15])
