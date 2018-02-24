#	Global variable goes here
#	first to define the dataset root

MPII_ROOT = "mpii/"
IMG_ROOT = MPII_ROOT + "images/"
ANNO_ROOT = MPII_ROOT + "train/"
TRAIN_LIST = MPII_ROOT + "train_list.txt"
GT_ROOT = MPII_ROOT + "gt/"

INPUT_SIZE = 368
base_lr = 4e-5
epoch = 600
batch_size = 16

LOGDIR = 'log/'