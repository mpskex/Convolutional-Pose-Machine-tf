#-*-coding:utf-8 -*-
import numpy as np

from dataset import Dataset
from CPM import CPM
import Global
from multiprocessing import Pool

dataset = Dataset(train_list_path=Global.TRAIN_LIST, img_root=Global.IMG_ROOT, anno_root=Global.ANNO_ROOT, gt_root=Global.GT_ROOT, batch_size=Global.batch_size, in_size=Global.INPUT_SIZE, pre_gen=True)

dataset.GenerateAllHeatMaps()
