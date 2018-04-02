#coding: utf-8

"""
    ICHIGO PROJ

    2018-03-26
    Liu Fangrui a.k.a. mpsk
    Beijing University of Technology
"""
import cv2
import numpy as np
import Global
import CPM
from skimage import transform, io


joint_num = 16
in_size = 368
out_size = 46

def resize_to_imgsz(hm, img):
    """ Create Tensor for joint position prediction

    :param  hm:     Assuming input of shape (sz, w, h, c)
    :param  img:    Assuming input of shape (sz, w, h, c)
    :return:
    """
    assert len(hm.shape) == 4 and len(img.shape) == 4
    ret_map = np.zeros((hm.shape[0], img.shape[1], img.shape[2], hm.shape[-1]), np.float32)
    for n in range(hm.shape[0]):
        for c in range(hm.shape[-1]):
            ret_map[n,:,:,c] = transform.resize(hm[n,:,:,c], img.shape[1:3])
    return ret_map

def joints_plot_image(joints, img, radius=3, thickness=2):
    """ Plot the joints on image

    :param joints:      (np.array)Assuming input of shape (joint_num, dim)
    :param img:         (image)Assuming
    :param radius:      (int)Radius
    :param thickness:   (int)Thickness
    :return:        RGB image
    """
    assert len(joints.shape)==3 and len(img.shape)==4
    colors = [(241,242,224), (196,203,128), (136,150,0), (64,77,0), 
            (201,230,200), (132,199,129), (71,160,67), (32,94,27),
            (130,224,255), (7,193,255), (0,160,255), (0,111,255),
            (220,216,207), (174,164,144), (139,125,96), (100,90,69),
            (252,229,179), (247,195,79), (229,155,3), (155,87,1),
            (231,190,225), (200,104,186), (176,39,156), (162,31,123),
            (210,205,255), (115,115,229), (80,83,239), (40,40,198)]
    ret = np.zeros(img.shape, np.uint8)
    for num in range(joints.shape[0]):
        for jnum in range(joints.shape[1]):
            print(tuple(joints[num, jnum].astype(int)))
            ret[num] = cv2.circle(img[num], (int(joints[num,jnum,1]), int(joints[num,jnum,0])), radius=radius, color=colors[jnum], thickness=thickness)
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
    joints = -1 * np.ones(shape=(hm.shape[0], joint_num, 2))
    weight = np.zeros(shape=(hm.shape[0], joint_num))
    for n in range(hm.shape[0]):
        for i in range(joint_num):
            index = np.unravel_index(hm[n, :, :, i].argmax(), hm.shape[1:3])
            if hm[n, index[0], index[1], i] > thresh:
                if coord == 'hm':
                    joints[n, i] = np.array(index)
                elif coord == 'img':
                    joints[n, i] = np.array(index) * (img.shape[1], img.shape[2]) / hm.shape[1:3]
                weight[n, i] = 1
    return joints, weight


def predict(img_list, model_path=None, thresh=0.05, is_name=False, cpu_only=True, model=None, id=0, debug=False):
    """ predict API
    You can input any size of image in this function. and the joint result is remaped to 
    the origin image according to each image's scale
    Just feel free to scale up n down :P

    Just be aware of the `is_name` param: this is to determine if
    
    :param img_list:    list of img (numpy array) !EVEN ONE PIC VAL ALSO NEED INPUT AS LIST!
    :param model_path:  path to load the model
    :param thresh:      threshold value (ignore some small peak)
    :param is_name:     define whether the input is name_list or numpy_list
    :param cpu_only:    CPU only mode or GPU accelerate mode
    :param model:       preload model to do the predict
    :return :
    """
    #   Assertion to check input format
    assert img_list != None and len(img_list) >= 1
    if model == None and model_path == None:
        print('[!]\tError!\tA mo(len(_img_list), pred_map.shape[-1]-1, 2)del or a model path must be given!')
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
                t_img = io.imread(img_list[idx])
                if t_img is None:
                    raise IOError
                _img_list.append(t_img)
            else:
                t_img = _img_list[idx]
            if debug == True:
                print('[*]\tt_img have shape of ', t_img.shape)
            input_list.append(cv2.resize(t_img, (368, 368)))
        except:
            print("[!]\tError!\tFailed to load image of index ", idx)
    #   convert list to numpy array
    _input = np.array(input_list)
    if debug:
        print('[*]\tInput have shape of ', _input.shape)
    
    #   using restore session
    if model == None:
        model = CPM.CPM(pretrained_model=None,
                    cpu_only=cpu_only,
                    training=False)
        model.BuildModel()
    if model_path is not None:
        model.restore_sess(model_path)

    #   get the last stage's result
    pred_map = model.sess.run(model.output, feed_dict={model.img: _input / 255.0})[:, -1]
    if debug:
        np.save('pred.npy', pred_map)

    j = -1 * np.ones((len(_img_list), pred_map.shape[-1]-1, 2))
    w = np.zeros((len(_img_list), pred_map.shape[-1]-1))
    for idx in range(len(_img_list)):
        #   re-project heatmap to origin size

        if debug:
            print("[*]\tpred map have shape of", pred_map.shape)
            print("[*]\timg have shape of", _img_list[idx].shape)
            
        r_pred_map = resize_to_imgsz(np.expand_dims(pred_map[idx],0), np.expand_dims(_img_list[idx],0))

        if debug:
            print("[*]\tresized pred map have shape of", r_pred_map.shape)
        #   predict joints pos

        j[idx], w[idx] = joints_pred_numpy(r_pred_map, np.expand_dims(_img_list[idx],0), thresh=thresh)
        if debug:
            #   visualize the joints with origin image
            ret_img = joints_plot_image(np.expand_dims(j[idx], 0), np.expand_dims(_img_list[idx],0), radius=10, thickness=5)
            v_pred_map = np.sum(r_pred_map, axis=(3))
            print("[*]\tpred visualized map have shape of ", v_pred_map.shape)
            
            io.imsave('vis.'+str(id)+'-'+str(idx)+'.jpg', np.sum(ret_img, axis=0))
            io.imsave('pred.'+str(id)+'-'+str(idx)+'.jpg', (np.sum(v_pred_map, axis=0)*255).astype(np.uint8))
    return j, w
    


if __name__=='__main__':
    """ Demo of Using the model API
    """
    img_names = ['test.1.png','test.2.png','test.3.png', 'test.4.png']
    #   input must be greater than one
    assert len(img_names) >= 1
    j = predict(img_names, 'model/model.ckpt-99', debug=True, is_name=True)
    print(j)