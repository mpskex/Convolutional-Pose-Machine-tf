#   Convolutional Pose Machine implemented with tensorflow

This model is still work in progress.

Trainable with MPI dataset.

Details are in [my blog](https://blog.csdn.net/mpsk07/article/details/79522809) in Chinese.

![CPM](https://raw.githubusercontent.com/mpskex/Convolutional-Pose-Machine-tf/master/demo/arch.png)

#   Model: Convolutional Pose Machine
Please go to my [site](http://mpskex.wicp.net/models) to choose the Convolutional Pose Machine model
And put the model in `model/` directory

#   Demo & Benchmark Results
![50FPS@GTX960 RealTime DEMO](https://raw.githubusercontent.com/mpskex/Convolutional-Pose-Machine-tf/master/demo/demo.jpg)
![PCKh Benchmark Result](https://raw.githubusercontent.com/mpskex/Convolutional-Pose-Machine-tf/master/demo/PCKh.png)
![Demo](https://raw.githubusercontent.com/mpskex/Convolutional-Pose-Machine-tf/master/demo/demo.jpg)

## Train & model update (2018-Nov)
*   Mobile CPM is avaliable! 50FPS!! try it out!
*   Parameters in Numpy is avaliable!
*   Native Data Generator in Paralleled structures
*   More Layers and Faster R-CNN Code in side (Find your gadget in repo)
*   more functionality! like layers (Proposal Layers, Dispatch Layers, RoI Align Layer) and Regularization stuffs
*   Pretrained model avaliable in days [here](http://mpskex.wicp.net/models)!

##  Live demo avaliable (2018-Jul)
*   Live Camera Demo 
*   Offline Video Processing Demo

![Live Demo](https://raw.githubusercontent.com/mpskex/Convolutional-Pose-Machine-tf/master/demo/live.gif)

#   TODOs
*   Still working in benchmark (Not in this project's workflow)

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
