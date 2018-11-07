import numpy as np
import tensorflow as tf

import sys
sys.path.append("..")
from net.Layers import LayerLibrary

if __name__ == '__main__':
    layer = LayerLibrary()
    sess = tf.Session()

    # want to crop 2x2 out of a 5x5 image, and resize to 4x4
    image = np.arange(3*25*5).astype('float32').reshape(3, 5, 5, 5)
    boxes = np.asarray([[1, 1, 3.2, 3], [1, 1, 3.2, 3], [1, 1, 3.2, 3]], dtype='float32')
    target = 4
    print np.arange(boxes.shape[0]).astype(np.int)
    #   np.arange(boxes.shape[0]).astype(np.int)
    #   np.zeros([boxes.shape[0]], dtype=np.int)
    ans = layer.roi_align(image, boxes, np.zeros([boxes.shape[0]], dtype=np.int), target)
    print sess.run(ans).shape
    """
    Expected values:
    4.5 5 5.5 6
    7 7.5 8 8.5
    9.5 10 10.5 11
    12 12.5 13 13.5
    """