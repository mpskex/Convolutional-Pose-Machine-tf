#-*-coding:utf-8 -*-
import sys
sys.path.append("..")

from net.MobileCPM import MobileCPM
import util.Global as Global

#   Thanks to wbenhibi@github
#   good datagen to use

in_size = 224
base_lr = 4e-5

_set='mpii'
print('--Creating Dataset')
if _set == 'mpii':
    import dataset.datagen_mpii as datagen
    dataset = datagen.DataGenerator(Global.joint_list, Global.IMG_ROOT, Global.training_txt_file, remove_joints=None, in_size=in_size)
elif _set == 'coco':
    import dataset.datagen_coco_multipro as datagen
    IMG_ROOT = "/var/data/COCO2017"
    COCO_anno_file = "/var/data/COCO2017/annotations/person_keypoints_train2017.json"
    COCO_anno_file_val = "/var/data/COCO2017/annotations/person_keypoints_val2017.json"
    dataset = datagen.DataGenerator(img_dir=IMG_ROOT, COCO_anno_file=COCO_anno_file, COCO_anno_file_val=COCO_anno_file_val, in_size=in_size)
dataset.generateSet(rand=True)
dataset = paral.DataGenParaWrapper(dataset, buff_size=4)

model = MobileCPM(base_lr=base_lr,
            joints=dataset.joints_list,
            in_size=in_size,
            batch_size=Global.batch_size,
            epoch=100,
            dataset=dataset,    
            log_dir="../model/mobileCPM/",
            pretrained_model='../model/mobilev1_1.0.npy')
model.BuildModel()
model.train()