#-*-coding:utf-8 -*-
import sys
sys.path.append("..")

from net.CPM import CPM
import util.Global as Global
import dataset.datagen_para_branch as paral
import dataset.datagen_mpii as datagen



_set = 'coco'
in_size = 368
base_lr = 4e-4

print('--Creating Dataset')
if _set == 'mpii':
    import dataset.datagen_mpii as datagen
    dataset = datagen.DataGenerator(Global.joint_list, Global.IMG_ROOT, Global.training_txt_file, remove_joints=None, in_size=in_size)
elif _set == 'coco':
    import dataset.datagen_coco as datagen
    IMG_ROOT = "/var/data/COCO2017"
    COCO_anno_file = "/var/data/COCO2017/annotations/person_keypoints_train2017.json"
    COCO_anno_file_val = "/var/data/COCO2017/annotations/person_keypoints_val2017.json"
    dataset = datagen.DataGenerator(img_dir=IMG_ROOT, COCO_anno_file=COCO_anno_file, COCO_anno_file_val=COCO_anno_file_val, in_size=in_size)
dataset.generateSet(rand=True)
dataset = paral.DataGenParaWrapper(dataset, buff_size=4)

model = CPM(base_lr=base_lr,
            joints=dataset.joints_list,
            in_size=in_size,
            batch_size=Global.batch_size,
            epoch=150,
            dataset = dataset,
            pretrained_model="../model/vgg19.npy",
            name='model/CPM_with_COCO_l2norm')
model.BuildModel()
model.train()
