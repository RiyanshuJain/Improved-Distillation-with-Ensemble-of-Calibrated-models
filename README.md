# Medical-Image-Analysis

## Overview
Data Augmentation has proved to be an effective method for classification of images. It significantly increases the diversity of data available for training our models, without actually collecting new data samples. In this project, we have looked on recently proposed augmentation techniques popularly known as MixUp, CutMix, AugMix and CutOut. We found that models trained using augmentation techniques are calibrated more better. Ensemble models consisting of deep Convolutional Neural Networks (CNN) have shown significant improvements in model generalization but at the cost of large computation and memory requirements. We try to distillate all of the augmentation techniques in different teacher models and combine the knowledge of all the well calibrated teacher models into one shallow well calibrated student model. 

## Calibration
1. `MixUp` :
In this augmentation technique, samples are generated during training by convexly combining random pairs of images and their associated labels. Hence, we can say that the classifier is trained not only on the training data, but also in the vicinity of each training sample.

2. `CutMix` :
CutMix is an augmentation strategy incorporating region-level replacement. For a pair of images, patches from one image are randomly cut and pasted onto the other image along with the ground truth labels being mixed together proportionally to the area of patches. CutMix replaces the removed regions with a patch from another image, which utilizes the fact that there is no uninformative pixel during training, making it more efficient and effective.

3. `CutOut` :
Cutout augmentation is a kind of regional dropout strategy in which a random patch from an image is zeroed out (replaced with black pixels). Cutout samples suffer from the decrease in information.

4. `AugMix` :
AugMix is a data processing technique which mixes randomly generated augmentations, improves model robustness and slots easily existing training pipelines. Augmix performs data mixing using the input image itself. It transforms (translate, shear, rotate and etc) the input image and mixes it with the original image. AugMix prevents degradation of images while maintaining diversity as a result of mixing the results of augmentation techniques in a convex combination.

## Contributors 
1. [Riyanshu Jain](https://github.com/RiyanshuJain)
2. [Divyam Patel](https://github.com/pateldivyam26)
3. [Dhruv Viradiya](https://github.com/DhruvViradiya1515)
