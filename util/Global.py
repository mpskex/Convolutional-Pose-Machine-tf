#	Global variable goes here
#	first to define the dataset root

IMG_ROOT = "/var/data/Dataset/images"
#IMG_ROOT = "/Users/mpsk/Documents/mpii/images/"

INPUT_SIZE = 368
base_lr = 2e-5
epoch = 150
batch_size = 16

LOGDIR = 'log/'

training_txt_file = "dataset.txt"
                                                                               
joint_list = ['r_anckle', 'r_knee', 'r_hip', 'l_hip', 'l_knee', 'l_anckle', 'pelvis', 'thorax', 'neck', 'head', 'r_wrist', 'r_elbow', 'r_shoulder', 'l_shoulder', 'l_elbow', 'l_wrist']
