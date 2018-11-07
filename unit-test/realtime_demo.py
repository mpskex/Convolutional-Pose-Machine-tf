#coding: utf-8

"""
    ICHIGO PROJ
    Real Time Mobile CPM Demo

    2018-04-11
    Liu Fangrui a.k.a. mpsk
    Beijing University of Technology
"""
import cv2
import sys
import time
import numpy as np

sys.path.append("..")
import util.predict as pdt
import net.MobileCPM as MobileCPM
import net.CPM as CPM

class RealTimeDemo(object):
    """ This is a Real Time Mobile CPM demo
    """
    def __init__(self, model):
        self.model = model

    def __draw_bbox__(self, img, bboxes, color=(255,0,0)):
        """ Draw Bounding box
        Args:
            img     :   input image
            bboxes  :   list of bounding boxes for this image
        Return:
            img     :   image with bbox visualization   
        """
        for bbox in bboxes:
            cv2.rectangle(img, (int(bbox[0] - bbox[2]//2), int(bbox[1] - bbox[3]//2)),
                          (int(bbox[0] + bbox[2]//2),
                           int(bbox[1] + bbox[3]//2)), color)
        return img

    def __estimate__(self, img):
        """ Estimate a img's result through a given model
        Note:   This function should be reload to fit other model's output

        Args:
            img         :   given_img
        Return:
            ret_img     :   visualized result
            t_pred      :   Prediction time stamp for calculating time cost
            t_postp     :   Post Processing time stamp for calculating time cost
        """
        j, w = pdt.predict([img], model=self.model, debug=False, is_name=False)
        t_pred = time.time()
        ret_img = pdt.joints_plot_image(np.expand_dims(j[0], 0), np.expand_dims(w[0],0), np.expand_dims(img,0), radius=5, thickness=5)[0]
        t_postp = time.time()
        return ret_img, t_pred, t_postp

    def run(self, debug=False):
        """ Run a Live demo
        """
        cap = cv2.VideoCapture(0)
        #   Fit the net's resolution to boost performance
        #   It's important
        cap.set(3,self.model.in_size)
        cap.set(4,self.model.in_size)
        while(cap.isOpened()):
            # get a frame
            ret, frame = cap.read()
            if ret:
                if debug:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    ret_img, t_pred, t_postp = self.__estimate__(frame)
                    cv2.imshow("Real Time Demo", cv2.cvtColor(ret_img, cv2.COLOR_RGB2BGR))
                else:
                    try:
                        # show a frame
                        t1 = time.time()
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        #   This is the general part of RT Demo
                        ret_img, t_pred, t_postp = self.__estimate__(frame)

                        print("Cost: predict %.2f ms. project %.2f ms."%((t_pred - t1)*1000, (t_postp-t_pred)*1000))
                        ret_img = cv2.putText(ret_img,
                                "Cost: %.2f ms  %.2f FPS. Press q to quit."%((t_pred-t1)*1000, 1/(t_pred-t1)), 
                                (10, 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 2)
                        cv2.imshow("Benchmark", cv2.cvtColor(ret_img, cv2.COLOR_RGB2BGR))
                    except Exception as e:
                        #wont interrupt RT demo
                        print("[!] ERROR occured!! "+str(e))

            else:
                print(ret)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__=='__main__':
    """ Performance Benchmark
    """

    # ''' >>>>>>> Old version
    #   NO NEED TO RESTORE!
    #   This is a better way to load flexible model cuz it wont create rebundant nodes
    model = MobileCPM.MobileCPM(pretrained_model='../model/mobileCPM_224/model.npy', load_pretrained=True, training=False, cpu_only=False,
                    stage=6, in_size=224)
    model.BuildModel()
    rtdemo = RealTimeDemo(model)
    rtdemo.run()
    #<<<<<<<<<<<<'''
