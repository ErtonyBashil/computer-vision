# Mask Detection Based on YOLOv5 pytorch


The project is a requirement due to complete the computer vision module at semester Dakar Institute of Technology 
the aim of the project is to annotate images in Roboflow and using an API to train the YOLOv5 model implemented pytorch
frame work on Google collaboratory.


![MaskDetection.gif](binary captured)


We used a custom dataset (the wearing mask dataset from Roboflow) to train a YOLOv5 model
using the TensorFlow framework in Google Colab.
The input can be an image or a video, and the output will be a detection of whether a mask is being worn or not."

YOLOv5 is a deep learning object detection developed by Jocher et al., (2020) using
PyTorch, thus, it can benefit from the ecosystem of PyTorch.  One of the biggest
reasons that make YOLOv5 popular is the productivity; YOLOv5 is very fast in terms
of processing speed, but even though it is speedy, it still can balance the accuracy, which makes it even better.

According to [ Integration of improved YOLOv5 for face mask detector and auto-labeling to generate dataset forfighting against COVID-19]
(https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9809528/) paper, by taking advantage of the PyTorch framework, the advantages of YOLOv5 models were a significantly smaller size, 
faster training time, and more accessibility to deployment in real-world applications.

We used manual labelization on Roboflow, we have performed random resized cropping and a vertical flip to each image dataset. Besides that, we also make
the Dataset square images with a usual pixel size of 224 x 224 since most deep neural networks, including ResNet, 
require square images as input. according to [YOLOv5 Based Real-time Helmet and Mask ](https://www.aasmr.org/liss/Vol.9/No.3%202022/Vol.9.No.3.08.pdf) paper

We used 3261 images as the Dataset, and the Dataset is divided into a training set, testing set, and validation set 
with a ratio of 6:2:2. The Dataset is pre-processed, such as rotation and zooming, to increase object detection 
performance.  And those images are labelled with ‘With_Mask’ and ‘Without_Mask’, and ‘Incorrect_Mask’. The
photos are divided into a training set with 682 illustrations, a testing set with 96 images, and a validation set with
85 images. This study using YOLOv5 as the training model. This study obtains the greatest result in 300 epochs instead
of 500 epochs, which prove that the higher epochs doesn’t always mean the greater performance.



### Implementation







according to [A YOVO5 Based Real-time Helmet and Mask ](https://www.aasmr.org/liss/Vol.9/No.3%202022/Vol.9.No.3.08.pdf) paper



 

Jian and Lang (2021) have created a face mask detection model based on
PaddlePaddle, You only look once (PP-YOLO) and enhanced the model with




transfer learning due to insufficient data samples

1. Set up the wokplace

    Add the daatset to google colab

2. Add the face-mask.yaml file to /yolov5/data/

4. Clone the yolov5 repo

5. Change your directory to the cloned


###  The result
Through these experiments, we observed
that the object background complexity will hugely affect the object detection result.
This study obtains a precision of 95% and 77% recall at the end of the study

This experiment contains 7959 images for the Dataset. In addition, this experiment uses Fine-tune as the transfer
learning strategy. They first pre-trained the PP-YOLO with the Dataset annotated in PascalVoc format, and a set of the Dataset with Mix-up data is sent to migration
training. This experiment obtains the highest 89.69% mAP by using PP-YOLO-mask. Through these experiments, we also can conclude that some enhancement strategy is
recommendable to enhance our model.

As mentioned before, this study has labelled the dataset into 5 classes, which are
"Head", "Helmet", "Incorrect Mask", "No Wearing Mask" and "Wearing Mask".
Consider the training environment and how to find the optimal batch size.
This study completed the model training with 3 different batch sizes. The training
per epochs in different batch sizes and the training loss are considerable values while Considering 
the time consumed and the accuracy and the training loss, this study found out that an epoch equal to 60 is the most suitable parameter for this study to
use. Even though precision resulted better in epochs 50, but the overall score shows
that the epochs 60 are consider good as well.

Difficulty while training the YOLOv5 model
This study has faced a major training environment problem. Due to the low-level
hardware, the training progress shall also consider some parameters that will affect
the hardware memory.


The Adam Optimizer using binary cross-entropy was
used to generate the model. The mark detection system was able to show excellent
result with accuracy of 97.8% with ResNet50.
















