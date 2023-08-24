# Corrosion-CNN-OD
This is a code focusing on the corrosion area location and corrosion degree detection of carbon steel materials.
# Requirements environment
PyTorch >=1.7.1 is recommended for the use of this code.
# Dataset preparation
1. This code was trained using the dataset in the VOC format.
   The tag files were placed in annotations under the VOCdevkit/VOC2007 folder.
   The image files were placed in JPEGImages under the VOCdevkit/VOC2007 folder.
3. voc_annotation.py was run to get 2007_train.txt and 2007_val.txt for training.
4. self_classes.txt was created to write the categories that will be distinguished.
# Training
Run train.py to start training. 
Classes_path is used in train.py to link to the self_classes.txt corresponding to the detection class.
# Testing
1. Modify model_path and classes_path in the yolo.py file.
   The model_path corresponds to the weight file, and the classes_path is the class to which model_path corresponds.
3. Run predict.py.

