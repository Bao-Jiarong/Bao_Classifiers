## All Nets Summary
The repository includes the implementation of VGG11, VGG13, VGG16,  VGG19, AlexNet, LeNet, ZFNet, Resnet18, Resnet34, Resnet50, Resnet101, Resnet152, SqueezeNet, SqueezeNetv2, SqueezeNetv3, MobileNet and MobileNetv2  in Tensorflow 2.  
I summerized all of the nets in the function Backbone, so that we wanna use different nets, instead of open one by one them from different folders, we can just give the name of net which we wanna use, for example, backbone_name ="VGG11".
And in the function Backbone, I made include_top variable, in case we don't need dense layers, we can set it into False to close dense layers.


### Training on MNIST
<center>
<img src="img/mnist.png" width="400" height="350"/>
</center>

### Requirement
```
python==3.7.0
numpy==1.18.1
```
### How to use
Training & Prediction can be run as follows:    
`python train.py train`  
`python train.py predict img.png`  

### Implementation Notes
* **Note 1**:   
Since some nets are somehow huge and painfully slow in training ,I decided to make number of filters variable. If you want to run it in your PC, you can change the number of filters into 64,32,16,8,4 or 2, according to the net you choose, beacause some nets only work well in a specified filters . For example:  
`model = src.backbone.Backbone((None,height, width, channel),classes = n_outputs,
filters = 8,
include_top = True,
backbone_name = backbone_name)`

* **Note 2** :   
You can also make the size of images smaller, so that it can be ran faster and doesn't take too much memories.

### Result for MNIST:
* width      = 224 >> 2  
* height     = 224 >> 2  
* Batch size = 32  
* epochs = 2  

note: the width and height in MobileNet are: 224 >> 1

Name |Learning Rate |  Fliters |  Optimizer  |  Training Accuracy |  Validation Accuracy  |
:---: | :---: | :---:|:---: | :---: | :---:
VGG11 | 0.0001 | 8|Adam | 92.74% | 97.13%
VGG13 | 0.0001 | 8|Adam | 92.00% | 96.04%
VGG16 | 0.0001 | 8|Adam | 90.98% | 96.98%
VGG19 | 0.0001 | 8|Adam | 89.13% | 96.63%
AlexNet| 0.0001 | 8|Adam | 91.79% | 96.28%
LeNet | 0.0001 | 8|Adam | 92.05% | 94.04%
ZFNet | 0.0001 | 8|Adam | 93.05% | 96.63%
ResNet18 | 0.0001 | 6|Adam | 85.52% | 92.29%
ResNet34 | 0.0001 | 6|Adam | 86.22% | 92.84%
ResNet50 | 0.0001 | 6|Adam | 92.83% | 94.28%
ResNet101 | 0.0001 | 6|Adam | 87.93% | 92.64%
ResNet152 | 0.0001 | 6|Adam | 94.46% | 95.62%
SqueezeNet | 0.0001 | 16|Adam | 88.65% | 95.47%
SqueezeNetv2 | 0.0001 | 16|Adam | 93.18% | 97.35%
SqueezeNetv3 | 0.0001 | 16|Adam | 93.77% | 97.39%
MobileNet | 0.002 | 64|Adam | 90.26% | 93.57%
MobileNetv2 | 0.0005 | 8|RMSprop | 90.23% | 95.92%
