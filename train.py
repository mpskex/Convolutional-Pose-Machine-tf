#-*-coding:utf-8 -*-
from CPM import CPM
import Global
'''
from dataset import Dataset
dataset = Dataset(train_list_path=Global.TRAIN_LIST, img_root=Global.IMG_ROOT, anno_root=Global.ANNO_ROOT, gt_root=Global.GT_ROOT, batch_size=Global.batch_size, in_size=Global.INPUT_SIZE, pre_gen=True)
#dataset.GenerateAllHeatMaps(procnum=12)
'''

#   Thanks to wbenhibi@github
#   good datagen to use

import datagen
print('--Creating Dataset')
dataset = datagen.DataGenerator(Global.joint_list, Global.IMG_ROOT, Global.training_txt_file, remove_joints=None, in_size=Global.INPUT_SIZE)
dataset._create_train_table()
dataset._randomize()
dataset._create_sets()
#'''

model = CPM(base_lr=Global.base_lr, in_size=Global.INPUT_SIZE, batch_size=Global.batch_size, epoch=Global.epoch, dataset = dataset, log_dir=Global.LOGDIR)
model.BuildModel()
model.train()

