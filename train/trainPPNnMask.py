#-*-coding:utf-8 -*-
import sys
sys.path.append("..")
import dataset.datagen_para_branch as paral
from net.PPNnMask import PPNnMask
import dataset.datagen_mpii_emb as datagen

#   Thanks to wbenhibi@github
#   good datagen to use

_set = 'mpii'
in_size = 368
base_lr = 4e-4
img_root = "/var/data/Dataset/images"
epoch = 150
batch_size = 16
inf_num = 16
training_txt_file = "../dataset.txt"                                                                       
joint_list = ['r_anckle', 'r_knee', 'r_hip', 'l_hip', 'l_knee', 'l_anckle', 'pelvis', 'thorax', 'neck', 'head', 'r_wrist', 'r_elbow', 'r_shoulder', 'l_shoulder', 'l_elbow', 'l_wrist']

#   PPNnMask V8

print('--Creating Dataset')
if _set == 'mpii':
    import dataset.datagen_mpii_emb_multipro as datagen
    dataset = datagen.DataGenerator(
        img_dir=img_root,
        train_data_file=training_txt_file,
        in_size=in_size,
        anchor_base_size=2,
        anchor_ratio=[1,2],
        anchor_scale=[1,4,8],
        inf_num=inf_num
        )
elif _set == 'coco':
    import dataset.datagen_coco_emb_multipro as datagen
    IMG_ROOT = "/var/data/COCO2017"
    COCO_anno_file = "/var/data/COCO2017/annotations/person_keypoints_train2017.json"
    COCO_anno_file_val = "/var/data/COCO2017/annotations/person_keypoints_val2017.json"
    dataset = datagen.DataGenerator(img_dir=IMG_ROOT,
        COCO_anno_file=COCO_anno_file,
        COCO_anno_file_val=COCO_anno_file_val,
        in_size=in_size,
        anchor_base_size=2,
        anchor_ratio=[1,2],
        anchor_scale=[1,4,8],
        inf_num=inf_num)
dataset.generateSet(rand=True)
dataset = paral.DataGenParaWrapper(dataset, buff_size=6)

model = PPNnMask(base_lr=base_lr,
            joints=dataset.joints_list,
            in_size=in_size,
            batch_size=batch_size,
            epoch=150,
            rois_max=16,
            dataset=dataset,
            anchors=dataset.anchors,
            pretrained_model= "../model/vgg19.npy",
            name="model/PPNv8@"+_set,
            _type='V8'
            )
model.BuildModel(no_mask_net=True)
model.train()
