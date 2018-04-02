#   Convolutional Pose Machine implemented with tensorflow

This model is still work in progress.

Trainable with MPI dataset.

Details are in [my blog](https://blog.csdn.net/mpsk07/article/details/79522809) in Chinese.

![CPM](http://img.blog.csdn.net/20180312094729995?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbXBzazA3/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

#   Model: Convolutional Pose Machine
Please go to my [site](http://mpskex.wicp.net/models) to choose the Convolutional Pose Machine model
And put the model in `model/` directory

#   Demo & Benchmark Results
![Demo](https://img-blog.csdn.net/20180402211711524?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L21wc2swNw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![PCKh Benchmark Result](https://img-blog.csdn.net/2018040221121922?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L21wc2swNw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

#   TODOs
*   Still working in benchmark (Not in this project's workflow)


#   Files
Filename|Usage
:-------|:----
predict.py|change the source file's image
train.py|use this to launch a training process
CPM.py|NO USAGE PLEASE LEAVE INTACT
datagen.py|NO USAGE PLEASE LEAVE INTACT

##  Credit
### Cited [Convolutional Pose Machine](https://arxiv.org/abs/1602.00134)
    @inproceedings{wei2016cpm,
        author = {Shih-En Wei and Varun Ramakrishna and Takeo Kanade and Yaser Sheikh},
        booktitle = {CVPR},
        title = {Convolutional pose machines},
        year = {2016}
    }

    Thanks to wbenbihi@github for his batch generator~

    Author: Liu Fangrui aka mpsk
        Beijing University of Technology
            College of Computer Science & Technology
