#-*-coding:utf-8 -*-
from dataset import Dataset
from CPM import CPM
import Global

dataset = Dataset(train_list_path=Global.TRAIN_LIST, img_root=Global.IMG_ROOT, anno_root=Global.ANNO_ROOT, gt_root=Global.GT_ROOT, batch_size=Global.batch_size, in_size=Global.INPUT_SIZE, pre_gen=True)
dataset.GenerateAllHeatMaps(procnum=12)

model = CPM(base_lr=Global.base_lr, in_size=Global.INPUT_SIZE, batch_size=Global.batch_size, epoch=Global.epoch, dataset = dataset, log_dir=Global.LOGDIR)
model.BuildModel()
model.train()

