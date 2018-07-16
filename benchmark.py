#coding: utf-8

"""
    ICHIGO PROJ

    2018-04-11
    Liu Fangrui a.k.a. mpsk
    Beijing University of Technology

    This script will record your stance and save
    the visualized pose prediction video
"""
import cv2
import time
import numpy as np
import predict as pdt
import CPM


if __name__=='__main__':
    """ Performance Benchmark
    """
    #''' <<<<<<< New version
    #   NO NEED TO RESTORE!
    model = CPM.CPM(pretrained_model='model/model.npy', load_pretrained=True, training=False, cpu_only=False,
                    stage=6)
    model.BuildModel()
    #>>>>>>>>>>>>'''

    ''' >>>>>>> Old version
    model = CPM.CPM(pretrained_model=None,
                    cpu_only=False,
                    training=False)
    model.BuildModel()
    model.restore_sess('IchigoBrain/model/model.ckpt-99')
    #<<<<<<<<<<<<'''

    cap = cv2.VideoCapture(0)
    cap.set(3,368)
    cap.set(4,368) 
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    out = None
    first = True
    while(cap.isOpened()):
        # get a frame
        ret, frame = cap.read()
        if ret:
            try:
                # show a frame
                t1 = time.time()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                j, w = pdt.predict([frame], model=model, debug=False, is_name=False)
                t_pred = time.time()
                ret_img = pdt.joints_plot_image(np.expand_dims(j[0], 0), np.expand_dims(frame,0), radius=5, thickness=5)[0]
                t_proj = time.time()
                print("Cost: predict %.2f ms. project %.2f ms."%((t_pred - t1)*1000, (t_proj-t_pred)*1000))
                ret_img = cv2.putText(ret_img,
                        "Cost: %.2f ms  %.2f FPS    Press q to quit."%((t_pred-t1)*1000, 1/(t_pred-t1)), 
                        (10, 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 2)
                ret_img = cv2.cvtColor(ret_img, cv2.COLOR_RGB2BGR)
                if first == True:
                    print(ret_img.shape[:-1][::-1])
                    out = cv2.VideoWriter('demo.mp4 ', fourcc, 5, ret_img.shape[:-1][::-1])
                    first = False
                if out is not None:
                    out.write(ret_img)
                cv2.imshow("Benchmark", ret_img)
            except Exception, e:
                print("[!] ERROR occured!! ", str(e))
        else:
            print(ret)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows() 