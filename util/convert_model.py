# coding: utf-8
import sys
sys.path.append("..")
import util.predict
import net.CPM

#   mpsk - 2018/04/19
#
#   Attention here
#   There's another way to load the model
#   which can speed up the loading process
#
#   Instead of using the ckpt(CheckPoint file)
#   We use NumPy binary file to restore the parameters
#   This can load parameters during constructing the graphs
#   No need of the initial value for NULL pretrained model
#
#   USAGE:
#       #   NO NEED TO RESTORE!
#       model = CPM.CPM(pretrained_model='model/test.npy', load_pretrained=True, training=False, cpu_only=False)
#       model.BuildModel()
#
#   I will upload the numpy model file upto my server this week.
#   ~Happy coding~

#   output model path
filename = "model/model.npy"

model = CPM.CPM(pretrained_model=None,
                cpu_only=False,
                training=False)
model.BuildModel()
model.restore_sess('model/model.ckpt-99')
model.save_npy(save_path=filename)