#coding: utf-8

import cv2
import numpy as np
import Global
import CPM

if __name__=='__main__':
    model = CPM.CPM(base_lr=Global.base_lr, in_size=Global.INPUT_SIZE, batch_size=Global.batch_size, epoch=Global.epoch, log_dir=Global.LOGDIR, 
                    #pretrained_model='log/model.npy', load_pretrained=True,
                    training=False)
    model.BuildModel()
    model.restore_sess('model/model.ckpt-49')
    
    frame = cv2.imread('test.1.png')

    frame = cv2.resize(frame, (368, 368))
    frame = np.expand_dims(frame, 0)
    print frame.shape
    frame = model.sess.run(model.output, feed_dict={model.img:frame/255.0})[-1]
    frame = np.expand_dims(np.sum(frame, axis=(0, 3)), 2)
    print frame
    cv2.imshow('capture', frame)
    cv2.waitKey()

    '''
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()

        frame = cv2.resize(frame, (368, 368))
        frame = model.sess.run(model.output, feed_dict={model.img:np.expand_dims(frame/255, 0)})[-1]
        frame = np.expand_dims(np.sum(frame, axis=(0,3)), 2)
        print frame.shape
        cv2.imshow('capture', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    '''