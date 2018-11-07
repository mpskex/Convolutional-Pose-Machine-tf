# coding: utf-8

"""
    ICHIGO PROJ
    Predict Utility

    2018-03-26
    Liu Fangrui a.k.a. mpsk
    Beijing University of Technology
"""
import time

import Global
import cv2
import numpy as np
from skimage import draw, io

import sys
sys.path.append("..")
import net.CPM
import net.MobileCPM

LINKS = [(0, 1), (1, 2), (2, 6), (6, 3), (3, 4), (4, 5), (6, 8),
         (8, 13), (13, 14), (14, 15), (8, 12), (12, 11), (11, 10)]

def get_model(model_type='MobileCPM', in_size=368, stage=6, npy_dir='IchigoBrain/model/'):
    """ Easy Get model API
    Args:
        model_type:     MobileCPM or CPM or other predefined model
        in_size:        Defines the input size of model 
                            cuz we only restore the params, so it's ok to change size
                            but may occur huge performance drop
        stage:          Select how many stages you want to use (model provided only have 6 stages)
    """
    print("[*]\tSelected %s model"%(model_type))
    if model_type == 'MobileCPM':
        model = MobileCPM.MobileCPM(pretrained_model=npy_dir+'mobile_model.npy', load_pretrained=True, training=False, cpu_only=False,
                                    stage=stage, in_size=in_size)
        model.BuildModel()
    elif model_type == 'CPM':
        model = CPM.CPM(pretrained_model=npy_dir+'model.npy', load_pretrained=True, training=False, cpu_only=False,
                        stage=stage, in_size=in_size)
        model.BuildModel()
    else:
        raise ValueError("No model type found named %s"%(model_type))
    return model

def get_mark(j_dt, j_gt, weight, metric='PCKh', t_thresh=0.2):
    """ Calculation of normalized distance

    !!! This function need the neck and head's presence !!!

    To tackle the scaling problem
    We used normalization which considers the distance between head and neck

    Also we have to deal with the shifting problem
    So we use the upright cordinates

    Args:
        j_dt:       predict joint array (joint_num x dimension)
        j_gt:       ground truth array  (joint_num x dimension)
        weight:     the presence of each joint
        metric:     PCK     -- normalized bt distance between right shoulder and right hip
                    PCKh    -- normalized by distance between head and neck

    """
    if type(j_dt) != np.ndarray:
        j_dt = np.array(j_dt)
    if type(j_gt) != np.ndarray:
        j_gt = np.array(j_gt)

    if metric == 'PCKh':
        normJ_a = 9
        normJ_b = 8
    elif metric == 'PCK':
        normJ_a = 12
        normJ_b = 3
    else:
        raise ValueError
    assert j_dt is not None and j_gt is not None
    assert j_dt.shape == j_gt.shape
    select_joint = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15]
    #   check presence
    for n in select_joint:
        if weight[n] != 1:
            select_joint.remove(n)

    #   Normalization
    err = -1 * np.ones(j_dt.shape[0], np.float)
    for j in [j_dt, j_gt]:
        for dim in range(j_dt.shape[-1]):
            norm_shift = j[:, dim].max()
            norm_scale = abs(j[normJ_a, dim] - j[normJ_b, dim])
            #   upright normalization
            j[:, dim] -= norm_shift
            #   scaling normalization by l1

    #   Select root node
    dist_list= []
    root_node = 8
    if (j_dt[8, :]!=-1).all() and (j_gt[8, :]!=-1).all():
        root_node = 8
    elif (j_dt[6, :]!=-1).all() and (j_gt[6, :]!=-1).all():
        root_node = 6
    else:
        #   find first non-zero weight position
        root_node = np.where(weight==1)[0]
    #   Calc dist
    for j in [j_dt, j_gt]:
        dist = np.zeros((len(select_joint)), np.float)
        public_norm = 1
        for idx in range(len(select_joint)):
            dist[idx] = np.linalg.norm(j[select_joint[idx]] - j[root_node]) / np.linalg.norm(j[normJ_a] - j[normJ_b])
        dist_list.append(dist)
        print(dist)
    err = np.abs(dist_list[0] - dist_list[1])

    # PCK / PCKh Error Calculation
    '''
    err[:] = weight * np.linalg.norm(j_gt[:, :] - j_dt[:, :], axis=1) / np.linalg.norm(j_gt[normJ_a, :] - j_gt[normJ_b, :], axis=0)
    '''
    #   err has shape of (joint_num) and every element is greater than 0 (if err[dim] == -1 then means it is not present)
    #   Using y = 1/kx (kx>=1) otherwise the mark is set to 1 (100%)
    #   To choose a proper k, we need to set a tolerence threshold like 0.2
    #   If we choose the threshlod 0.2, then the k = 1/threshold (k=5)
    print(err)
    mark = np.ones(err.shape, np.float)
    aver_mark = 0.0
    dim = 0
    k = 1 / t_thresh
    for dim in range(err.shape[0]):
        if k * err[dim] >= 0:
            if k * err[dim] <= 1:
                mark[dim] = 1
            elif k * err[dim] > 1:
                mark[dim] = 1 / (k * err[dim])
            aver_mark += mark[dim]
            dim += 1
        else:
            #   if mark[dim] == 0 then we skip it to compute the mark
            mark[dim] = 0
    return aver_mark / len(select_joint)


def resize_to_imgsz(hm, img):
    """ Create Tensor for joint position prediction

    :param  hm:     Assuming input of shape (sz, w, h, c)
    :param  img:    Assuming input of shape (sz, w, h, c)
    :return:
    """
    assert len(hm.shape) == 4 and len(img.shape) == 4
    ret_map = np.zeros(
        (hm.shape[0], img.shape[1], img.shape[2], hm.shape[-1]), np.float32)
    for n in range(hm.shape[0]):
        for c in range(hm.shape[-1]):
            '''
            ret_map[n, :, :, c] = transform.resize(
                hm[n, :, :, c], img.shape[1:3])
            '''
            ret_map[n, :, :, c] = cv2.resize(hm[n, :, :, c], img.shape[1:3][::-1])
    return ret_map


def joints_name_image(joints, img, letters, radius=3, thickness=2):
    """ Plot the joints on image

    :param joints:      (np.array)Assuming input of shape (joint_num, dim)
    :param img:         (image)Assuming
    :param radius:      (int)Radius
    :param thickness:   (int)Thickness
    :return:        RGB image
    """
    assert len(joints.shape) == 3 and len(img.shape) == 4
    assert joints.shape[0] == img.shape[0]
    colors = [(241, 242, 224), (196, 203, 128), (136, 150, 0), (64, 77, 0),
              (201, 230, 200), (132, 199, 129), (71, 160, 67), (32, 94, 27),
              (130, 224, 255), (7, 193, 255), (0, 160, 255), (0, 111, 255),
              (220, 216, 207), (174, 164, 144), (139, 125, 96), (100, 90, 69),
              (252, 229, 179), (247, 195, 79), (229, 155, 3), (155, 87, 1),
              (231, 190, 225), (200, 104, 186), (176, 39, 156), (162, 31, 123),
              (210, 205, 255), (115, 115, 229), (80, 83, 239), (40, 40, 198)]
    ret = np.zeros(img.shape, np.uint8)
    for num in range(joints.shape[0]):
        for jnum in range(joints.shape[1]):
            ret[num] = cv2.putText(img[num], letters[jnum], (int(joints[num, jnum, 1]), int(
                joints[num, jnum, 0])), cv2.FONT_HERSHEY_SIMPLEX, .6, color=colors[jnum], thickness=2)
    return ret


def joints_plot_image(joints, weight, img, radius=3, thickness=2):
    """ Plot the joints on image

    :param joints:      (np.array)Assuming input of shape (num, joint_num, dim)
    :param img:         (image)Assuming input of shape (num, w, h, c)
    :param weight:      (np.array)Assuming input of shape (num, joint_num)
    :param radius:      (int)Radius
    :param thickness:   (int)Thickness
    :return:            set of RGB image (num, w, h, c)
    """
    assert len(joints.shape) == 3 and len(img.shape) == 4 and len(weight.shape) == 2
    assert joints.shape[0] == img.shape[0] == weight.shape[0]
    colors = [(241, 242, 224), (196, 203, 128), (136, 150, 0), (64, 77, 0),
              (201, 230, 200), (132, 199, 129), (71, 160, 67), (32, 94, 27),
              (130, 224, 255), (7, 193, 255), (0, 160, 255), (0, 111, 255),
              (220, 216, 207), (174, 164, 144), (139, 125, 96), (100, 90, 69),
              (252, 229, 179), (247, 195, 79), (229, 155, 3), (155, 87, 1),
              (231, 190, 225), (200, 104, 186), (176, 39, 156), (162, 31, 123),
              (210, 205, 255), (115, 115, 229), (80, 83, 239), (40, 40, 198)]
    ret = np.zeros(img.shape, np.uint8)
    assert len(joints.shape) == 3 and len(img.shape) == 4
    assert img.shape[-1] == 3
    ret = img.copy()
    for num in range(joints.shape[0]):
        for jnum in range(joints.shape[1]):
            if weight[num, jnum] == 1:
                rr, cc = draw.circle(
                    int(joints[num, jnum, 0]), int(joints[num, jnum, 1]), radius)
                ret[num, rr, cc] = colors[jnum]
    for num in range(joints.shape[0]):
        for lnk in range(len(LINKS)):
            if weight[num, LINKS[lnk][0]] == 1 and weight[num, LINKS[lnk][1]] == 1:
                rr, cc = draw.line(int(joints[num, LINKS[lnk][0], 0]), int(joints[num, LINKS[lnk][0], 1]),
                                int(joints[num, LINKS[lnk][1], 0]), int(joints[num, LINKS[lnk][1], 1]))
                ret[num, rr, cc] = colors[lnk]
    return ret


def joints_pred_numpy(hm, img, coord='img', thresh=0.2):
    """ Create Tensor for joint position prediction

    :param  hm:     Assuming input of shape (sz, w, h, c)
    :param  img:    Assuming input of shape (sz, w, h, c)
    :param  coord:  project to original image or not
    :param  thresh: Threshold to limit small respond
    :return:
    """
    assert len(hm.shape) == 4 and len(img.shape) == 4
    joints = -1 * np.ones(shape=(hm.shape[0], hm.shape[3] - 1, 2))
    weight = np.zeros(shape=(hm.shape[0], hm.shape[3] - 1))
    for n in range(hm.shape[0]):
        for i in range(hm.shape[3] - 1):
            index = np.unravel_index(hm[n, :, :, i].argmax(), hm.shape[1:3])
            if hm[n, index[0], index[1], i] > thresh:
                if coord == 'hm':
                    joints[n, i] = np.array(index)
                elif coord == 'img':
                    joints[n, i] = np.array(
                        index) * (img.shape[1], img.shape[2]) / hm.shape[1:3]
                weight[n, i] = 1
    return joints, weight


def predict(img_list, thresh=0.2, is_name=False, cpu_only=True, model=None, id=0, debug=False):
    """ predict API
    You can input any size of image in this function. and the joint result is remaped to
    the origin image according to each image's scale
    Just feel free to scale up n down :P

    Just be aware of the `is_name` param: this is to determine if

    :param img_list:    list of img (numpy array) !EVEN ONE PIC VAL ALSO NEED INPUT AS LIST!
    :param thresh:      threshold value (ignore some small peak)
    :param is_name:     define whether the input is name_list or numpy_list
    :param cpu_only:    CPU only mode or GPU accelerate mode
    :param model:       preload model to do the predict
    :return :
    """
    t = time.time()
    #   Assertion to check input format
    assert model != None
    assert img_list != None and len(img_list) >= 1
    if model == None:
        print('[!]\tError!\tA model or a model path must be given!')
        raise ValueError
    if is_name:
        assert type(img_list[0]) == str
    if is_name == True:
        _img_list = []
    else:
        _img_list = img_list
    input_list = []
    for idx in range(len(img_list)):
        try:
            if is_name == True:
                t_img = io.imread(img_list[idx])[:, :, :3]
                if t_img is None:
                    raise IOError
                _img_list.append(t_img)
            else:
                t_img = _img_list[idx]
            if debug == True:
                print('[*]\tt_img have shape of ', t_img.shape)
            '''
            #   Histogram Equalization
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            for ch in range(t_img.shape[-1]):
                t_img[:,:,ch] = clahe.apply(t_img[:,:,ch])
            cv2.imshow('debug', cv2.cvtColor(t_img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            '''
            t_img = cv2.resize(t_img, (model.in_size, model.in_size))
            input_list.append(cv2.resize(t_img, (model.in_size, model.in_size)))
        except Exception as e:
            print("[!]\tError!\tFailed to load image of index ", idx)
            print(e)
    # convert list to numpy array
    _input = np.array(input_list)
    if debug:
        print('[*]\tInput have shape of ', _input.shape)

    #   get the last stage's result
    if debug:
        t1 = time.time()
    pred_map = model.get_joint_map(_input / 255.0)[:, -1]

    if debug:
        t2 = time.time()
        print("[*] Net Cost : %.2f ms, CheckPoint : %.2f ms"%((t2-t1)*1000, (t2-t)*1000))
        np.save('pred.npy', pred_map)

    j = -1 * np.ones((len(_img_list), pred_map.shape[-1] - 1, 2))
    w = np.zeros((len(_img_list), pred_map.shape[-1] - 1))
    #   Gaussian Blur to predict map
    for n in range(pred_map.shape[0]):
        for ch in range(pred_map.shape[-1]):
            pred_map[n, :, :, ch] = cv2.GaussianBlur(pred_map[n, :, :, ch], (5, 5), 0.5)
    '''
    '''
    if debug:
        t2 = time.time()
        print("[*] \tArrayAlloc & Gaussian Blur CheckPoint : %.2f ms"%((t2-t)*1000))
    for idx in range(len(_img_list)):
        #   re-project heatmap to origin size

        if debug:
            print("[*]\tpred map have shape of", pred_map.shape)
            print("[*]\timg have shape of", _img_list[idx].shape)
            t2 = time.time()
            print("[*] \tIn_Iter CheckPoint : %.2f ms"%((t2-t)*1000))

        r_pred_map = resize_to_imgsz(np.expand_dims(
            pred_map[idx], 0), np.expand_dims(_img_list[idx], 0))

        if debug:
            t2 = time.time()
            print("[*] \tResize CheckPoint : %.2f ms"%((t2-t)*1000))
            print("[*]\tresized pred map have shape of", r_pred_map.shape)

        #   predict joints pos
        j[idx], w[idx] = joints_pred_numpy(
            r_pred_map, np.expand_dims(_img_list[idx], 0), thresh=thresh)
        if debug:
            t2 = time.time()
            print("[*] \tArgmax Checkpoint : %.2f ms"%((t2-t)*1000))
            #   visualize the joints with origin image
            ret_img = joints_plot_image(np.expand_dims(j[idx], 0), np.expand_dims(w[idx], 0), np.expand_dims(_img_list[idx], 0), radius=10, thickness=5)
            v_pred_map = np.sum(r_pred_map, axis=(3))
            print("[*]\tpred visualized map have shape of ", v_pred_map.shape)

            io.imsave('vis.' + str(id) + '-' + str(idx) +
                      '.jpg', np.sum(ret_img, axis=0))
            io.imsave('pred.' + str(id) + '-' + str(idx) + '.jpg',
                      (np.sum(v_pred_map, axis=0) * 150).astype(np.uint8))
    return j, w


if __name__ == '__main__':
    """ Demo of Using the model API
    """
    img_names = ['test.1.png', 'test.2.png', 'test.3.png', 'test.4.png']
    #   input must be greater than one
    assert len(img_names) >= 1
    j, w = predict(img_names, 'model/model.ckpt-99', debug=True, is_name=True)
    score = get_mark(j[0], j[0], np.logical_and(w[0], w[0]), t_thresh=0.2)
    print(j)
