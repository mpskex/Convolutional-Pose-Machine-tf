# coding: utf-8
import cv2
import h5py
import numpy as np

def __parse_txt(self, txt_name, debug=False):
        """
            How to generate Dataset:
            Create a TEXT file with the following structure:
                image_name.jpg[LETTER] box_xmin box_ymin box_xmax b_ymax joints
                [LETTER]:
                    One image can contain multiple person. To use the same image
                    finish the image with a CAPITAL letter [A,B,C...] for 
                    first/second/third... person in the image
                joints : 
                    Sequence of x_p y_p (p being the p-joint)
                    /!\ In case of missing values use -1
            RETURN:
                records(
                    record[name](
                        box, joints
                    )
                    , ...
                )
                ? * 2 * ?
        """
        try:
            records = []
            re_dict = {}
            with open(txt_name, "r") as f:
                records = f.readlines()
                f.close()
            for n in range(len(records)):
                records[n] = records[n].split(' ')
                records[n][-1] = records[n][-1].split('\r\n')[0]
                name = records[n][0]
                box = np.reshape(np.array(records[n][1:5]),
                    (2,2), order='C')
                joints = np.reshape(np.array(records[n][5:]),
                    (16,2), order='C')
                re_dict[name] = {box, joints}
                if debug:
                    print box.shape, "\t", joints.shape
                    print records[n]
            return re_dict
        except Exception, e:
            raise e

if __name__ == "__main__":
    parse_txt("dataset.txt", debug=True)
