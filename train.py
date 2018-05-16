#-*-coding:utf-8 -*-
from CPM import CPM
import Global
import datagen_mpii as datagen

#   Thanks to wbenhibi@github
#   good datagen to use

_set='coco'

print('--Creating Dataset')
if _set == 'mpii':
    import datagen_mpii as datagen
    dataset = datagen.DataGenerator(Global.joint_list, Global.IMG_ROOT, Global.training_txt_file, remove_joints=None, in_size=Global.INPUT_SIZE)
elif _set == 'coco':
    import datagen_coco as datagen
    IMG_ROOT = "/var/data/COCO2017"
    COCO_anno_file = "/var/data/COCO2017/annotations/person_keypoints_train2017.json"
    COCO_anno_file_val = "/var/data/COCO2017/annotations/person_keypoints_val2017.json"
    dataset = datagen.DataGenerator(img_dir=IMG_ROOT, COCO_anno_file=COCO_anno_file, COCO_anno_file_val=COCO_anno_file_val, in_size=Global.INPUT_SIZE)
dataset.generateSet(rand=True)

model = CPM(base_lr=Global.base_lr, joints=dataset.joints_list, in_size=Global.INPUT_SIZE, batch_size=Global.batch_size, epoch=Global.epoch, dataset = dataset, log_dir=Global.LOGDIR)
model.BuildModel()
model.restore_sess(model="log.150_epoch_coco/model.ckpt-149")
model.train()
