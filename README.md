# Some Training Tips of Deep Learning Model for The Beginners

This project summarizes some tips for training a deep learning model for image classification, in which the model is hand-crafted from scratch based on the Keras library. The experiments is evaluated on the Kaggle dataset [''dog breed identification''](https://www.kaggle.com/c/dog-breed-identification). Due to the computational limitations of the google colaboratory, only 5 classes with the top 5 samples is used for the classification. [Here is the source code of demo.](https://github.com/joy030631/DL_training_tips.git)

The training tips are concluded as follows.
(1) The random seed for the initialization of deep network is fixed for reproducibility of the experiment.

(2) The experiments upon various epoches and the mini-batch size are examined to validate the criterion as proposed by [Yoshua Bengio](https://arxiv.org/abs/1206.5533) and [Luschi et al](https://arxiv.org/abs/1804.07612). Our experimental results are consists with these works.

(3) Few strategies to prevent over- and underfitting are concluded and validated by the experiment; in which the dropout layer is validated and demonstrate a better performance of the model with dropout.

(4) We utilize the pre-trained VGG16, as the state-of-art model for the image classification, to further improve the performance.

see more detail in section ["Training tips"](https://hackmd.io/@B3ulMYzHRrO6CWIJxb_HOw/SyBebkF0B)
