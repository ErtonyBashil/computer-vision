# Mask Detection Based on YOLOv5 pytorch


The project is a requirement for completing the Computer Vision module at Dakar Institute of Technology.
The aim of the project is to annotate images using Roboflow and then utilize an API to train the YOLOv5 
model, implementing PyTorch framework, on Google Colaboratory


![Demo gif](test_data/MaskDetection.gif)


We used a custom dataset (the wearing mask dataset from Roboflow) to train a YOLOv5 model
using the TensorFlow framework in Google Colab.
The input can be an image or a video, and the output will be a detection of whether a mask is being worn or not."

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
<hr>


transfer learning due to insufficient data samples

1. Clone Repo and install all dependencies

```python
!git clone https://github.com/ultralytics/yolov5  # clone repo
%cd yolov5
%pip install -qr requirements.txt # install dependencies
%pip install -q roboflow

import torch
import os
from IPython.display import Image, clear_output  # to display images

print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")
```


**Step 2.  Install roboflow and import dataset using roboflow API**

```python
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="-------------------")
project = rf.workspace("ditnov202").project("face_mask_detection-wfniz")
dataset = project.version(1).download("yolov5")
```


**Step 3: Train Our Custom YOLOv5 model**

```python
!python train.py --img 640 --batch 80 --epochs 256 --data {dataset.location}/data.yaml --weights yolov5s.pt --cache
```

**4. Run the inference with the trained weight**

```python
!python detect.py --weights /content/yolov5/runs/train/exp3/weights/best.pt --img 640 --conf 0.1 --source /content/yolov5/data/images/zidane.jpg

```
**5. Display inference on ALL test images**

```python
import glob
from IPython.display import Image, display

for imageName in glob.glob('/content/yolov5/runs/detect/exp4/*.jpg'): #assuming JPG
    display(Image(filename=imageName))
    
!python detect.py --weights /content/yolov5/runs/train/exp3/weights/best.pt  --conf 0.25 --source '/content/Test_data'

#export the model's weights for future use
from google.colab import files
files.download('./runs/train/exp3/weights/best.pt')
```

![zidane Mask detection](test_data/zidane.jpg) ![Multiple Mask detection](test_data/multiple.jpg)
![Joselyne](test_data/Joselyne.jpg)

###  The result
<hr>
Through these experiments, we observedthat the object background complexity will hugely affect the object detection result.
This study obtains a precision of 94% and 77% recall at the end of the project

This experiment contains 7959 images for the Dataset. As mentioned before, this study has labelled the dataset into 3 classes, which are
"masque", "pas de masque", "masque mal porte".

Batch : 80
Epochs : 256
image size : 640

We found out that an epoch equal to 300 and 120 iteration was the most suitable parameter for this project
However we faced a major training environment problem due to the low-level
hardware granted on collab. Therefore, We trained with 250 epochs and 80 iterations
use. Even though precision resulted better in epochs 80, but the overall score shows that the epochs 120 are consider good as well.


The Adam Optimizer using binary cross-entropy was used to generate the model. 
The mark detection system was able to show excellent result with accuracy of 97.8%.

During the annotation we chose some pictures with no head as null values.
We first found out that core success of the project lied on the labelization. No matter how you ajust the parameters
is the annotation is not done well, the score turns to be awfull.













