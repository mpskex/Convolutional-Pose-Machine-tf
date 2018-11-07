# coding: utf-8

"""
    ICHIGO PROJ
    Offline Vidio processing demo with smoothing

    2018-04-11
    Liu Fangrui a.k.a. mpsk
    Beijing University of Technology
"""
import cv2
import time
import numpy as np
import sys

sys.path.append("..")
import util.predict as pdt
import net.CPM as CPM
import net.MobileCPM as MobileCPM

if __name__ == '__main__':
    """ Performance Benchmark
    """
    #''' >>>>>>> Old version
    #   NO NEED TO RESTORE!
    #   This is a better way to load flexible model cuz it wont create rebundant nodes
    model = MobileCPM.MobileCPM(pretrained_model='IchigoBrain/model/mobile_model.npy', load_pretrained=True, training=False, cpu_only=False,
                    stage=6, in_size=224)
    model.BuildModel()
    #<<<<<<<<<<<<'''

    for vid_name in ['demo.train.avi', 'demo.old.avi']:
        print("-- Proceeding video ", vid_name)
        cap = None
        out = None
        fourcc = None
        cap = cv2.VideoCapture(vid_name)
        fourcc = cv2.VideoWriter_fourcc(*'MPEG')
        first = True
        parrallel_num = 16
        smooth_factor = 0.8
        j_t = None
        w_t = None
        j_smooth = None
        while (cap.isOpened()):
            # get a frame
            # if True:
            try:
                # show a frame
                t1 = time.time()
                batch_frame = []
                for n in range(parrallel_num):
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        batch_frame.append(frame)
                    else:
                        print(ret)
                        break
                j, w = pdt.predict(batch_frame, model_in_size=model.in_size, model=model, debug=False, is_name=False)
                j = np.array(j)
                w = np.array(w)
                t_pred = time.time()
                if j_smooth is None:
                    j_smooth = np.zeros(j.shape, np.float)
                for n in range(parrallel_num):
                    if j_t is None or w_t is None:
                        j_smooth[n] = j[n]
                        j_t = np.zeros(j.shape[1:], np.float)
                        w_t = np.zeros(j.shape[1:2], np.float)
                    else:
                        j_smooth[n, np.where((w_t == 0) & (w[n] == 1))] = j[n, np.where((w_t == 0) & (w[n] == 1))]
                        j_smooth[n, np.where((w_t == 1) & (w[n] == 1))] = (1 - smooth_factor) * j[
                            n, np.where((w_t == 1) & (w[n] == 1))].astype(np.float) + smooth_factor * j_t[
                                                                              np.where((w_t == 1) & (w[n] == 1))]
                    j_t[np.where(w[n] == 1)] = j_smooth[n, np.where(w[n] == 1)]
                    w_t = w[n]
                t_smos = time.time()
                ret_img = pdt.joints_plot_image(j_smooth, w, np.array(batch_frame), radius=5, thickness=5)
                t_proj = time.time()
                print("Cost: predict %.2f ms. smoothing %.2f ms projection %.2f ms." % (
                (t_pred - t1) * 1000, (t_smos - t_pred) * 1000, (t_proj - t_smos) * 1000))
                '''
                ret_img = cv2.putText(ret_img,
                        "Cost: %.2f ms  %.2f FPS    Press q to quit."%((t_pred-t1)*1000, 1/(t_pred-t1)), 
                        (10, 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 2)
                '''
                for n in range(parrallel_num):
                    ret_img_s = cv2.cvtColor(ret_img[n], cv2.COLOR_RGB2BGR)
                    if first == True:
                        print(ret_img_s.shape[:-1][::-1])
                        out = cv2.VideoWriter('out_' + str(smooth_factor * 10) + '_' + vid_name, fourcc, 30,
                                              ret_img_s.shape[:-1][::-1])
                        first = False
                    if out is not None:
                        out.write(ret_img_s)
                # cv2.imshow("Benchmark", ret_img)
                # '''
            except Exception as e:
                print("[!] ERROR occured!! ", str(e))
                # '''
        cap.release()
        out.release()