#coding: utf-8
import h5py
import numpy as np
import scipy.io

"""
This tool is to convert pretrained mat to numpy format
Model donwnloaded from MatConvNet can easily transform to npy file
Numpy file is native to my model structure(inspired by VGG19.npy)

Liu Fangrui a.k.a. mpsk
Beijing University of Technology
"""
conv_layer = ['filter', 'bias']
bn_layer = ['mult', 'bias', 'moments']

def ReadMat(mat_name, _type='ori'):
    """ Load mat to mem

    :param mat_name:    name (path) to mat
    :param _type:       `ori` for legacy `hdf5` for h5py mode
    :return:            mat object
    """
    if _type == 'ori':
        return scipy.io.loadmat(mat_name)
    elif _type == 'hdf5':
        return h5py.File(mat_name, 'r')

def ParamtoDict(mat, _type='matconv', debug=False):
    """ Convert params to dict

    :param mat:         mat object
    :param _type:       mode to load default to mat conv net structure
    """
    assert mat is not None
    d = {}
    layers = []
    if _type == 'matconv':
        print('mat object have ', mat['params'].size, ' params.')
        for i in range(mat['params'].size):
            indx = mat['params'][0][i]
            '''
            pname = indx[0][0].split('_')[-1]
            lname = ""
            for p in range(len(indx[0][0].split('_')[:-1])):
                if p != len(indx[0][0].split('_')[:-1])-1:
                    lname += indx[0][0].split('_')[p]+ '_'
                else:
                    lname += indx[0][0].split('_')[p]
            if debug:
                print 'para have name ', pname, ' and belongs to ', lname
            '''
            print 'converting', indx[0][0], ' ...'
            d[indx[0][0]] = indx[1]
        return d
    else:
        raise NotImplementedError

def GetParaName(mat, _type='matconv'):
    """ Get all parameters name

    :param mat:     mat object
    :param _type:   mode for loading
    :return:        parameter
    """

if __name__ == '__main__':
    mat = ReadMat('resnet50.mat', _type='ori')
    ret = ParamtoDict(mat, debug=True)
    np.save('resnet50.npy', ret)