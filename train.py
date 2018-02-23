#-*-coding:utf-8 -*-
from dataset import Dataset
from CPM import CPM
import Global

dataset = Dataset(Global.TRAIN_LIST, Global.IMG_ROOT, Global.ANNO_ROOT, Global.batch_size, Global.INPUT_SIZE)

model = CPM(base_lr=Global.base_lr, in_size=Global.INPUT_SIZE, batch_size=Global.batch_size, epoch=Global.epoch, dataset = dataset, img_root=Global.IMG_ROOT, log_dir=Global.LOGDIR)
model.BuildModel()
#model.train()

